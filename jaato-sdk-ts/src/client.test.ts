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
  readyState: number;
  binaryType: string;
  onopen: (() => void) | null;
  onmessage: ((msg: { data: string }) => void) | null;
  onerror: (() => void) | null;
  onclose: ((info: { code: number; reason: string }) => void) | null;
  emit(event: object): void;
  emitClose(code?: number, reason?: string): void;
  send(text: string): void;
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
      send(text: string): void {
        this.sent.push(text);
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
});

describe("JaatoClient event stream", () => {
  beforeEach(() => installMockWebSocket());
  afterEach(() => restoreWebSocket());

  test("subscribe receives events emitted from the server", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    const received: JaatoEvent[] = [];
    client.subscribe((e) => received.push(e));
    await connectAndAck(client);

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
    const unsub = client.subscribe((e) => received.push(e));
    await connectAndAck(client);
    unsub();
    lastInstance!.emit({
      type: EventTypeValue.SYSTEM_MESSAGE,
      timestamp: new Date().toISOString(),
    } as unknown as JaatoEvent);
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
    assert.equal(received.length, 0);
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
