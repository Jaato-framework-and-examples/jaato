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
let reads = 0;
let onCommand: (args: string[]) => void = () => undefined;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => true,
  getClient: () => ({
    serverProtocolVersion: "1.31",
    fetchWorkspaceFile: async () => (reads++, manifest === null)
      ? { event: { ok: false, category: "not_found" }, data: null }
      : { event: { ok: true }, data: new TextEncoder().encode(JSON.stringify(manifest)) },
    executeCommand: async (_c: string, args: string[]) => { commands.push(args); onCommand(args); },
    stageFiles: async (_w: string, files: Array<{ name: string }>) => ({ staged: files.map((f) => f.name), failed: [] }),
  }),
  reassertAfterReconnect: async () => undefined,
}));

const { EnvironmentPanel, proposalText, stripState } = await import("./EnvironmentPanel");

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
    expect(await screen.findByText("Node.js 22 detected in api/.nvmrc.")).toBeInTheDocument();
    expect(screen.getByTestId("environment-strip")).toHaveTextContent("1 suggestion waiting");
    // The rail badge counts it with the section closed.
    await waitFor(() => expect(useJaato.getState().environmentWaiting).toBe(1));
    expect(screen.getByText("api/AGENTS.md")).toBeInTheDocument();
    expect(commands).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(commands).toEqual([["bind", "node", "22"]]));
    // A finished install is "✓ just now" on its tile, not a banner.
    expect(await screen.findByText("✓ just now")).toBeInTheDocument();
    expect(screen.queryByText(/installed\./)).toBeNull();
    expect(screen.getByTestId("environment-strip")).toHaveTextContent("1 bound · ready");
    expect(useJaato.getState().environmentWaiting).toBe(0);
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
    expect(await screen.findByText("Installing Node.js 22…")).toBeInTheDocument();
    expect(screen.getByTestId("environment-job")).toHaveTextContent("$ mise install node@22");
    // While installing, everything but Cancel is locked.
    expect(screen.getByRole("button", { name: "Rescan repositories" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Add toolchain/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel install" }));
    await waitFor(() => expect(commands).toEqual([["cancel"]]));
    expect(await screen.findByRole("button", { name: "Retry node" })).toBeInTheDocument();
  });

  it("shows the mid-session hint and binds from it", async () => {
    inSession(true);
    manifest = { toolchains: [], proposals: [], guidance: [], job: null };
    const done = vi.fn();
    render(<EnvironmentPanel url="./api/environment" workspace={WS} hint={{ command: "npx", tool: "node" }} onHintDone={done} fetchImpl={fakeFetch().fetchImpl} pollMs={20} />);
    expect(await screen.findByTestId("environment-hint")).toHaveTextContent("“npx” was not found in the last command. Node.js 22 provides it.");
    fireEvent.click(screen.getByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(done).toHaveBeenCalled());
    await waitFor(() => expect(commands).toEqual([["bind", "node", "22"]]));
  });
});

describe("EnvironmentPanel after it is gone", () => {
  it("stops re-reading the manifest once unmounted mid-command", async () => {
    inSession(true);
    manifest = { toolchains: [], proposals: [PROPOSAL], guidance: [], job: null };
    const view = render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fakeFetch().fetchImpl} pollMs={40} />);
    fireEvent.click(await screen.findByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(commands).toEqual([["bind", "node", "22"]]));
    view.unmount();
    const atUnmount = reads;
    await new Promise((r) => setTimeout(r, 150));
    expect(reads - atUnmount).toBeLessThanOrEqual(1);
  });
});

describe("the status strip", () => {
  const label = (t: string) => ({ node: "Node.js", java: "Java" } as Record<string, string>)[t] ?? t;
  const job = (status: string, extra: Record<string, unknown> = {}) => ({
    id: "j", action: "bind", tool: "java", version: "21", status, log: [], error: null, notes: [], startedAt: "t", finishedAt: null, ...extra,
  }) as never;
  const base = { job: null, label, waiting: 0, live: true, bound: 0, pending: 0 };

  it("derives one state, first match wins", () => {
    expect(stripState({ ...base, job: job("running"), waiting: 2 })).toMatchObject({ tone: "running", text: "Installing Java 21…", action: "cancel" });
    expect(stripState({ ...base, job: job("failed", { error: "no mise" }), waiting: 2 })).toMatchObject({ tone: "failed", text: "Java 21 failed: no mise", action: "retry" });
    expect(stripState({ ...base, job: job("cancelled") })).toMatchObject({ tone: "failed", text: "Install of Java 21 cancelled." });
    expect(stripState({ ...base, waiting: 2, bound: 1 })).toMatchObject({ tone: "waiting", text: "2 suggestions waiting" });
    expect(stripState({ ...base, live: false, pending: 2 })).toMatchObject({ tone: "neutral", text: "2 chosen · bound when the first session starts" });
    expect(stripState({ ...base })).toMatchObject({ tone: "neutral", text: "No toolchains bound" });
    expect(stripState({ ...base, job: job("done"), bound: 1 })).toMatchObject({ tone: "ready", text: "1 bound · ready", action: null });
  });
});

describe("EnvironmentPanel with no session yet", () => {
  it("remembers a choice instead of installing, and says when it will be bound", async () => {
    inSession(false);
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fakeFetch().fetchImpl} pollMs={20} />);
    expect(await screen.findByRole("note")).toHaveTextContent("bound when the first session in this workspace starts");
    expect(screen.queryByRole("button", { name: "Rescan repositories" })).toBeNull();
    const add = screen.getByRole("button", { name: /Add toolchain/ });
    expect(add).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(add);
    expect(screen.getByRole("radio", { name: "22" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("radio", { name: "20" }));
    fireEvent.click(screen.getByRole("radio", { name: "22" }));
    fireEvent.click(screen.getByRole("button", { name: "Choose Node.js" }));
    expect(await screen.findByTestId("environment-pending")).toHaveTextContent("Node.js22");
    expect(screen.getByTestId("environment-strip")).toHaveTextContent("1 chosen · bound when the first session starts");
    expect(screen.queryByRole("radiogroup")).toBeNull();
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
