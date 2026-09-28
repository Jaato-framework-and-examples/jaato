/**
 * The Toolchains panel (#1344).  In a session it reads the plugin's
 * ``.jaato/environment.json`` through the daemon and acts by sending the
 * ``toolchain`` command; without one (the New workspace plate) a choice is
 * remembered and bound when the first session starts.  Pinned here: nothing
 * installs without a click, a bind is a command and never a backend call,
 * "Not now" goes to the backend, and a pending choice is sent once a
 * session exists.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useJaato } from "@/store/store";

const WS = "/srv/ws/demo";
const BASE = { workspace: WS, declined: [] as string[], allowed: [{ tool: "node", label: "Node.js", versions: ["22", "20"], server: null }] };
const PROPOSAL = { tool: "node", label: "Node.js", version: "22", pin: "22", pinAllowed: true, source: "api/.nvmrc" };

let manifest: Record<string, unknown> | null = null;
const commands: string[][] = [];
let onCommand: (args: string[]) => void = () => undefined;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => true,
  getClient: () => ({
    serverProtocolVersion: "1.31",
    fetchWorkspaceFile: async () => manifest === null
      ? { event: { ok: false, category: "not_found" }, data: null }
      : { event: { ok: true }, data: new TextEncoder().encode(JSON.stringify(manifest)) },
    executeCommand: async (_c: string, args: string[]) => { commands.push(args); onCommand(args); },
    stageFiles: async (_w: string, files: Array<{ name: string }>) => ({ staged: files.map((f) => f.name), failed: [] }),
  }),
  reassertAfterReconnect: async () => undefined,
}));

const { EnvironmentPanel, proposalText } = await import("./EnvironmentPanel");

function fakeFetch(handler: (url: string, init?: RequestInit) => unknown = () => BASE) {
  const calls: Array<{ url: string; body?: unknown }> = [];
  const f = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), body: init?.body ? JSON.parse(String(init.body)) : undefined });
    return new Response(JSON.stringify(handler(String(url), init) ?? {}), { status: 200, headers: { "content-type": "application/json" } });
  });
  return { calls, fetchImpl: f as unknown as typeof fetch };
}

function inSession(on: boolean): void {
  useJaato.setState((s) => ({
    sessionId: on ? "s1" : null,
    workspace: { ...s.workspace, selected: "demo", list: [{ name: "demo", path: WS } as never] },
  }));
}

beforeEach(() => {
  manifest = null; commands.length = 0; onCommand = () => undefined;
  localStorage.clear();
});
afterEach(() => { cleanup(); inSession(false); });

describe("the toolchain chip comes from the daemon", () => {
  it("says when the repository pins a version the server does not offer", () => {
    expect(proposalText({ tool: "go", label: "Go", version: "1.23", pin: "1.19", pinAllowed: false, source: "svc/go.mod" }))
      .toBe("Go detected (svc/go.mod pins 1.19, which this server does not offer). Bind 1.23 instead?");
  });

  const notice = (data: Record<string, unknown>) => ({
    type: "tool.result_enriched", agent_id: "main", call_id: "c1", tool_name: "cli_based_tool",
    plugin: "toolchain_offer", kind: "toolchain_offer", data,
  });

  it("a toolchain_offer notice becomes the rail's hint, only with a backend", () => {
    const st = useJaato.getState();
    st.setEnvironmentUrl(null);
    st.dispatch([notice({ command: "javac", tool: "java", bound: null })] as never);
    expect(useJaato.getState().environmentHint).toBeNull();
    st.setEnvironmentUrl("./api/environment");
    st.dispatch([notice({ command: "javac", tool: "java", bound: null })] as never);
    expect(useJaato.getState().environmentHint).toEqual({ command: "javac", tool: "java" });
    st.setEnvironmentHint(null);
    st.setEnvironmentUrl(null);
  });

  it("the page detects nothing itself: a not-found line with no notice raises no chip", () => {
    const st = useJaato.getState();
    st.setEnvironmentUrl("./api/environment");
    st.dispatch([
      { type: "tool.call_start", agent_id: "main", call_id: "c2", tool_name: "cli_based_tool", tool_args: {} },
      { type: "tool.output", agent_id: "main", call_id: "c2", chunk: "bash: line 1: go: command not found" },
      { type: "tool.call_end", agent_id: "main", call_id: "c2", success: true },
    ] as never);
    expect(useJaato.getState().environmentHint).toBeNull();
    // Nor does a notice for a toolchain that is already bound, or of another kind.
    st.dispatch([notice({ command: "go", tool: "go", bound: "1.23" })] as never);
    st.dispatch([{ ...notice({ command: "go", tool: "go", bound: null }), kind: "something_else" }] as never);
    expect(useJaato.getState().environmentHint).toBeNull();
    st.setEnvironmentUrl(null);
  });
});

describe("EnvironmentPanel in a session", () => {
  it("shows the plugin's proposals and binds with the toolchain command, never a backend call", async () => {
    inSession(true);
    manifest = { toolchains: [], proposals: [PROPOSAL], guidance: ["api/AGENTS.md"], job: null };
    onCommand = (args) => {
      if (args[0] === "bind") manifest = { ...manifest, proposals: [], toolchains: [{ tool: "node", version: "22", bin: ["node"], boundAt: "t", server: null }],
        job: { id: "j1", action: "bind", tool: "node", version: "22", status: "done", log: [], error: null, notes: [], startedAt: "t", finishedAt: "t" } };
    };
    const { calls, fetchImpl } = fakeFetch();
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fetchImpl} pollMs={20} />);
    expect(await screen.findByText("Node.js 22 detected (from api/.nvmrc). Bind it?")).toBeInTheDocument();
    expect(screen.getByText("api/AGENTS.md")).toBeInTheDocument();
    expect(commands).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(commands).toEqual([["bind", "node", "22"]]));
    expect(await screen.findByText("node 22 installed.")).toBeInTheDocument();
    const unbind = await screen.findByRole("button", { name: "Unbind node" });
    await waitFor(() => expect(unbind).not.toBeDisabled());
    fireEvent.click(unbind);
    await waitFor(() => expect(commands.at(-1)).toEqual(["unbind", "node"]));
    expect(calls.some((c) => /\/(bind|unbind)$/.test(c.url))).toBe(false);
  });

  it("Not now declines the proposal through the backend", async () => {
    inSession(true);
    manifest = { toolchains: [], proposals: [PROPOSAL], guidance: [], job: null };
    let declined = false;
    const { calls, fetchImpl } = fakeFetch((url) => {
      if (url.endsWith("/decline")) { declined = true; return {}; }
      return declined ? { ...BASE, declined: ["node"] } : BASE;
    });
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fetchImpl} pollMs={20} />);
    fireEvent.click(await screen.findByRole("button", { name: "Not now" }));
    await waitFor(() => expect(screen.queryByTestId("environment-proposal")).toBeNull());
    expect(calls.find((c) => c.url.endsWith("/decline"))?.body).toEqual({ workspace: WS, tool: "node" });
  });

  it("follows a running install, offers Cancel, and Retry once it failed", async () => {
    inSession(true);
    const job = { id: "j2", action: "bind", tool: "node", version: "22", status: "running", log: ["$ mise install node@22"], error: null, notes: [], startedAt: "t", finishedAt: null };
    manifest = { toolchains: [], proposals: [], guidance: [], job };
    onCommand = (args) => { if (args[0] === "cancel") manifest = { ...manifest, job: { ...job, status: "cancelled" } }; };
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fakeFetch().fetchImpl} pollMs={20} />);
    expect(await screen.findByText("Installing node 22…")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel install" }));
    await waitFor(() => expect(commands).toEqual([["cancel"]]));
    expect(await screen.findByRole("button", { name: "Retry node" })).toBeInTheDocument();
  });

  it("shows the mid-session hint and binds from it", async () => {
    inSession(true);
    manifest = { toolchains: [], proposals: [], guidance: [], job: null };
    const done = vi.fn();
    render(<EnvironmentPanel url="./api/environment" workspace={WS} hint={{ command: "npx", tool: "node" }} onHintDone={done} fetchImpl={fakeFetch().fetchImpl} pollMs={20} />);
    expect(await screen.findByTestId("environment-hint")).toHaveTextContent("npx was not found in the last command.");
    fireEvent.click(screen.getByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(done).toHaveBeenCalled());
    await waitFor(() => expect(commands).toEqual([["bind", "node", "22"]]));
  });
});

describe("EnvironmentPanel with no session yet", () => {
  it("remembers a choice instead of installing, and says when it will be bound", async () => {
    inSession(false);
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fakeFetch().fetchImpl} pollMs={20} />);
    expect(await screen.findByRole("note")).toHaveTextContent("bound when the first session in this workspace starts");
    fireEvent.change(screen.getByRole("combobox", { name: "Toolchain" }), { target: { value: "node" } });
    fireEvent.click(screen.getByRole("button", { name: "Choose" }));
    expect(await screen.findByTestId("environment-pending")).toHaveTextContent("node 22");
    expect(commands).toEqual([]);
    const { pendingBinds } = await import("@/app/toolchains");
    expect(pendingBinds(WS)).toEqual([{ tool: "node", version: "22" }]);
  });

  it("the pending choice is bound once a session starts, and leaves the list when its job ends", async () => {
    const { addPending, flushPending, pendingBinds } = await import("@/app/toolchains");
    addPending(WS, "node", "22");
    inSession(true);
    manifest = { toolchains: [], proposals: [], guidance: [], job: null };
    onCommand = (args) => {
      if (args[0] === "bind") manifest = { ...manifest, toolchains: [{ tool: "node", version: "22", bin: [], boundAt: "t", server: null }],
        job: { id: "j9", action: "bind", tool: "node", version: "22", status: "done", log: [], error: null, notes: [], startedAt: "t", finishedAt: "t" } };
    };
    await act(async () => { await flushPending(WS, 10); });
    expect(commands).toEqual([["bind", "node", "22"]]);
    expect(pendingBinds(WS)).toEqual([]);
  });

  it("a pending bind the plugin refuses is given up, not waited on forever", async () => {
    const mod = await import("@/app/toolchains");
    mod.addPending(WS, "node", "18");
    inSession(true);
    manifest = { toolchains: [], proposals: [], guidance: [], job: null };
    vi.useFakeTimers();
    const run = mod.flushPending(WS, 1000);
    await vi.advanceTimersByTimeAsync(mod.START_GRACE_MS + 5000);
    await run;
    vi.useRealTimers();
    expect(commands).toEqual([["bind", "node", "18"]]);
    expect(mod.pendingBinds(WS)).toEqual([]);
  });
});
