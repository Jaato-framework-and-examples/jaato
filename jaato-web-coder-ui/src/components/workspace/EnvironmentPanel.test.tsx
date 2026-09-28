/**
 * The Toolchains panel (#1344): a proposal installs only on a click, "Not
 * now" is sent to the backend (which remembers it), a running install can be
 * cancelled, and a failed one can be retried.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { EnvironmentPanel } from "./EnvironmentPanel";
import { proposalText } from "@/app/environment";
import { useJaato } from "@/store/store";

const WS = "/srv/ws/demo";
const BASE = {
  workspace: WS, toolchains: [], declined: [], job: null, guidance: ["api/AGENTS.md"],
  allowed: [{ tool: "node", label: "Node.js", versions: ["22", "20"], server: null }],
  proposals: [{ tool: "node", label: "Node.js", version: "22", pin: "22", pinAllowed: true, source: "api/.nvmrc" }],
};

function fakeFetch(handler: (url: string, init?: RequestInit) => unknown) {
  const calls: Array<{ url: string; method?: string; body?: unknown }> = [];
  const f = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), method: init?.method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    return new Response(JSON.stringify(handler(String(url), init) ?? {}), { status: 200, headers: { "content-type": "application/json" } });
  });
  return { calls, fetchImpl: f as unknown as typeof fetch };
}

afterEach(() => cleanup());

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

describe("EnvironmentPanel", () => {
  it("scans on mount in clone-time mode, and binds only on a click", async () => {
    let job = { id: "j1", tool: "node", version: "22", status: "running", log: ["$ mise install node@22"], notes: [], startedAt: "t" };
    const { calls, fetchImpl } = fakeFetch((url) => {
      if (url.endsWith("/refresh")) return { status: BASE };
      if (url.endsWith("/bind")) return { job };
      if (url.includes("/jobs/j1")) { job = { ...job, status: "done" }; return { job }; }
      // The backend reports the workspace's latest job beside the bindings.
      return { ...BASE, proposals: [], job, toolchains: [{ tool: "node", version: "22", bin: ["node"], boundAt: "t", server: null }] };
    });
    render(<EnvironmentPanel url="./api/environment" workspace={WS} scan fetchImpl={fetchImpl} />);
    expect(await screen.findByText("Node.js 22 detected (from api/.nvmrc). Bind it?")).toBeInTheDocument();
    expect(calls.filter((c) => c.url.endsWith("/bind"))).toHaveLength(0);
    expect(screen.getByText("api/AGENTS.md")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(calls.find((c) => c.url.endsWith("/bind"))?.body).toEqual({ workspace: WS, tool: "node", version: "22" }));
    expect(await screen.findByText("node 22 installed.", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Unbind node" })).toBeInTheDocument();
  });

  it("Not now declines the proposal through the backend", async () => {
    let declined = false;
    const { calls, fetchImpl } = fakeFetch((url) => {
      if (url.endsWith("/decline")) { declined = true; return {}; }
      return declined ? { ...BASE, proposals: [], declined: ["node"] } : BASE;
    });
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fetchImpl} />);
    fireEvent.click(await screen.findByRole("button", { name: "Not now" }));
    await waitFor(() => expect(screen.queryByTestId("environment-proposal")).toBeNull());
    expect(calls.find((c) => c.url.endsWith("/decline"))?.body).toEqual({ workspace: WS, tool: "node" });
  });

  it("a running install offers Cancel; a failed one says why and offers Retry", async () => {
    const running = { id: "j2", tool: "node", version: "22", status: "running", log: [], notes: [], startedAt: "t" };
    const { calls, fetchImpl } = fakeFetch((url) => {
      if (url.endsWith("/cancel")) return { job: { ...running, status: "cancelled" } };
      return { ...BASE, proposals: [], job: running };
    });
    render(<EnvironmentPanel url="./api/environment" workspace={WS} fetchImpl={fetchImpl} />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel install" }));
    expect(await screen.findByRole("button", { name: "Retry node" })).toBeInTheDocument();
    expect(calls.some((c) => c.url.endsWith("/jobs/j2/cancel"))).toBe(true);
  });

  it("shows the mid-session hint and binds from it", async () => {
    const { calls, fetchImpl } = fakeFetch((url) => {
      if (url.endsWith("/bind")) return { job: { id: "j3", tool: "node", version: "22", status: "done", log: [], notes: [], startedAt: "t" } };
      return { ...BASE, proposals: [] };
    });
    const done = vi.fn();
    render(<EnvironmentPanel url="./api/environment" workspace={WS} hint={{ command: "npx", tool: "node" }} onHintDone={done} fetchImpl={fetchImpl} />);
    expect(await screen.findByTestId("environment-hint")).toHaveTextContent("npx was not found in the last command.");
    fireEvent.click(screen.getByRole("button", { name: "Bind Node.js" }));
    await waitFor(() => expect(done).toHaveBeenCalled());
    expect(calls.find((c) => c.url.endsWith("/bind"))?.body).toEqual({ workspace: WS, tool: "node", version: "22" });
  });
});
