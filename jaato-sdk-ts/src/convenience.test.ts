// Convenience facade (Session.ask / complete / stream / permissions / dispose).
//
// Drives Session with a duck-typed fake client so the contracts are exercised
// deterministically without a daemon: the no-hang invariant (a plain turn emits
// only TURN_COMPLETED), source filtering, typed completion, streaming, the
// permission fail-loud path, error raising, and AsyncDisposable.

import { strict as assert } from "node:assert";
import { test } from "node:test";

import { JaatoClient } from "./client.js";
import { Session, type ClientToolHandler } from "./convenience.js";
import { AgentError, PermissionUnhandled } from "./errors.js";
import { EventTypeValue } from "./events.js";

type Handler = (ev: unknown) => void;

/** Records calls; sendMessage fires a scripted [eventType, event] sequence into
 *  the synchronously-dispatched handlers. */
class FakeClient {
  handlers = new Map<string, Handler[]>();
  calls: Array<[string, ...unknown[]]> = [];
  sessionId: string | null = "sid-1";
  #fire: Array<[string, unknown]>;

  constructor(fire: Array<[string, unknown]> = []) {
    this.#fire = fire;
  }

  subscribe(eventType: string, handler: Handler): () => void {
    const list = this.handlers.get(eventType) ?? [];
    list.push(handler);
    this.handlers.set(eventType, list);
    return () => {
      const l = this.handlers.get(eventType);
      if (l) this.handlers.set(eventType, l.filter((h) => h !== handler));
    };
  }
  subscribeOnce(eventType: string, handler: Handler): () => void {
    return this.subscribe(eventType, handler);
  }
  async sendMessage(
    text: string,
    attachments?: unknown,
    parallelTools?: unknown,
  ): Promise<void> {
    this.calls.push(["sendMessage", text, attachments, parallelTools]);
    for (const [eventType, ev] of this.#fire) {
      for (const h of [...(this.handlers.get(eventType) ?? [])]) h(ev);
    }
  }
  async respondToPermission(requestId: string, response: string): Promise<void> {
    this.calls.push(["respondToPermission", requestId, response]);
  }
  async respondToToolExecution(callId: string, result = "", error = ""): Promise<void> {
    this.calls.push(["respondToToolExecution", callId, result, error]);
  }
  async registerClientTools(tools: unknown): Promise<void> {
    this.calls.push(["registerClientTools", tools]);
  }
  async close(): Promise<void> {
    this.calls.push(["close"]);
  }
}

function session(fire: Array<[string, unknown]> = []): { s: Session; c: FakeClient } {
  const c = new FakeClient(fire);
  const s = new Session(c as unknown as JaatoClient);
  return { s, c };
}

const out = (source: string, text: string): [string, unknown] => [
  EventTypeValue.AGENT_OUTPUT,
  { source, text },
];
const TURN: [string, unknown] = [EventTypeValue.TURN_COMPLETED, {}];

// ── ask ──────────────────────────────────────────────────────────

test("ask: plain turn (only TURN_COMPLETED) does not hang, returns model text", async () => {
  const { s } = session([out("model", "Hello"), TURN]);
  assert.equal(await s.ask("hi"), "Hello");
});

test("ask: filters to model source by default", async () => {
  const { s } = session([out("model", "A"), out("tool", "[t]"), out("system", "[s]"), TURN]);
  assert.equal(await s.ask("hi"), "A");
});

test("ask: sources null collects everything", async () => {
  const { s } = session([out("model", "A"), out("tool", "B"), TURN]);
  assert.equal(await s.ask("hi", { sources: null }), "AB");
});

test("ask: forwards parallelTools + attachments to sendMessage", async () => {
  const { s, c } = session([out("model", "ok"), TURN]);
  const att = [{ mime_type: "image/png", data: "..", display_name: "x.png" }];
  await s.ask("hi", { parallelTools: false, attachments: att });
  const send = c.calls.find((x) => x[0] === "sendMessage")!;
  assert.deepEqual(send[2], att);
  assert.equal(send[3], false);
});

test("ask: raises AgentError on an error terminal", async () => {
  const { s } = session([
    [EventTypeValue.SESSION_TERMINATED, { reason: "error", error_type: "APIError", error_summary: "boom" }],
  ]);
  await assert.rejects(() => s.ask("hi"), (e: unknown) => {
    assert.ok(e instanceof AgentError);
    assert.equal((e as AgentError).errorType, "APIError");
    assert.equal((e as AgentError).errorSummary, "boom");
    return true;
  });
});

// ── complete ─────────────────────────────────────────────────────

test("complete: returns the typed payload", async () => {
  const { s } = session([
    [EventTypeValue.AGENT_COMPLETED, { payload: { name: "Alice", age: 30 } }],
    [EventTypeValue.SESSION_TERMINATED, { reason: "natural" }],
  ]);
  assert.deepEqual(await s.complete("Alice is 30."), { name: "Alice", age: 30 });
});

test("complete: no completion returns null (no hang)", async () => {
  const { s } = session([TURN]);
  assert.equal(await s.complete("hi"), null);
});

// ── stream ───────────────────────────────────────────────────────

test("stream: yields chunks then stops", async () => {
  const { s } = session([out("model", "Hel"), out("model", "lo"), TURN]);
  const chunks: string[] = [];
  for await (const c of s.stream("hi")) chunks.push(c);
  assert.deepEqual(chunks, ["Hel", "lo"]);
});

test("stream: raises AgentError on error terminal", async () => {
  const { s } = session([
    [EventTypeValue.SESSION_TERMINATED, { reason: "error", error_type: "X", error_summary: "y" }],
  ]);
  await assert.rejects(async () => {
    for await (const _ of s.stream("hi")) void _;
  }, AgentError);
});

// ── permissions ──────────────────────────────────────────────────

test("permission: unhandled denies then makes the turn raise PermissionUnhandled", async () => {
  const c = new FakeClient([TURN]);
  const s = new Session(c as unknown as JaatoClient); // no onPermission
  // simulate a gated tool firing during the session
  for (const h of c.handlers.get(EventTypeValue.PERMISSION_REQUESTED) ?? []) {
    h({ request_id: "r1", tool_name: "delete_file" });
  }
  // let the async permission handler settle
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(c.calls.some((x) => x[0] === "respondToPermission" && x[1] === "r1" && x[2] === "n"));
  await assert.rejects(() => s.ask("hi"), (e: unknown) => {
    assert.ok(e instanceof PermissionUnhandled);
    assert.equal((e as PermissionUnhandled).toolName, "delete_file");
    return true;
  });
});

test("permission: callback (sync + async) is used for the response", async () => {
  const c1 = new FakeClient();
  new Session(c1 as unknown as JaatoClient, () => "y");
  for (const h of c1.handlers.get(EventTypeValue.PERMISSION_REQUESTED) ?? []) {
    h({ request_id: "r1", tool_name: "t" });
  }
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(c1.calls.some((x) => x[0] === "respondToPermission" && x[2] === "y"));

  const c2 = new FakeClient();
  new Session(c2 as unknown as JaatoClient, async () => "a");
  for (const h of c2.handlers.get(EventTypeValue.PERMISSION_REQUESTED) ?? []) {
    h({ request_id: "r2", tool_name: "t" });
  }
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(c2.calls.some((x) => x[0] === "respondToPermission" && x[2] === "a"));
});

// ── host-tool dispatch ───────────────────────────────────────────

function withTools(
  handlers: Record<string, ClientToolHandler>,
): { s: Session; c: FakeClient } {
  const c = new FakeClient();
  const map = new Map(Object.entries(handlers));
  const s = new Session(c as unknown as JaatoClient, undefined, map);
  return { s, c };
}

function fireToolExec(c: FakeClient, ev: unknown): void {
  for (const h of [...(c.handlers.get(EventTypeValue.TOOL_EXECUTE_REQUEST) ?? [])]) h(ev);
}

test("host tool: handler runs client-side; object result is JSON-encoded back", async () => {
  let seen: unknown;
  const { c } = withTools({
    get_weather: (args) => {
      seen = args;
      return { weather: "sunny" };
    },
  });
  fireToolExec(c, { call_id: "c1", tool_name: "get_weather", tool_args: { city: "Paris" } });
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(seen, { city: "Paris" });
  assert.ok(
    c.calls.some(
      (x) => x[0] === "respondToToolExecution" && x[1] === "c1" && x[2] === '{"weather":"sunny"}',
    ),
  );
});

test("host tool: string result passed through verbatim", async () => {
  const { c } = withTools({ echo: () => "hi" });
  fireToolExec(c, { call_id: "c2", tool_name: "echo", tool_args: {} });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(c.calls.some((x) => x[0] === "respondToToolExecution" && x[1] === "c2" && x[2] === "hi"));
});

test("host tool: unknown tool name → error response", async () => {
  const { c } = withTools({ known: () => "x" });
  fireToolExec(c, { call_id: "c3", tool_name: "mystery", tool_args: {} });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(
    c.calls.some((x) => x[0] === "respondToToolExecution" && x[1] === "c3" && x[3] !== ""),
  );
});

test("host tool: handler throw → error response", async () => {
  const { c } = withTools({
    boom: () => {
      throw new Error("kaboom");
    },
  });
  fireToolExec(c, { call_id: "c4", tool_name: "boom", tool_args: {} });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(
    c.calls.some((x) => x[0] === "respondToToolExecution" && x[1] === "c4" && x[3] === "kaboom"),
  );
});

test("no tool handlers → no TOOL_EXECUTE_REQUEST subscription wired", () => {
  const { c } = session();
  assert.equal((c.handlers.get(EventTypeValue.TOOL_EXECUTE_REQUEST) ?? []).length, 0);
});

// ── lifecycle ────────────────────────────────────────────────────

test("close and Symbol.asyncDispose both close the underlying client", async () => {
  const { s, c } = session();
  await s.close();
  await s[Symbol.asyncDispose]();
  assert.equal(c.calls.filter((x) => x[0] === "close").length, 2);
});

test("client getter exposes the underlying client for high/low mixing", () => {
  const { s, c } = session();
  assert.equal(s.client, c as unknown as JaatoClient);
});
