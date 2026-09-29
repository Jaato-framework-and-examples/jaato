/**
 * The Files panel's finder (protocol 1.32): locate any workspace file,
 * changed or not, hidden or not.
 *
 * What the component must get right: it asks only once the query is long
 * enough, a stale answer never replaces a newer query's, a file the panel
 * hides is still found (and marked), a credential file offers no download,
 * and a walk that stopped early says so.  Drawn only for a daemon that
 * serves the verb.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useJaato } from "@/store/store";

const search = vi.fn();
vi.mock("@/sdk/connection", async (orig) => ({
  ...(await orig<typeof import("@/sdk/connection")>()),
  getClient: () => ({ searchWorkspaceFiles: search }),
}));
const download = vi.fn(async (_p: string) => undefined);
vi.mock("@/app/downloads", async (orig) => ({
  ...(await orig<typeof import("@/app/downloads")>()),
  downloadFromPanel: (p: string) => download(p),
}));

const { FileFinder } = await import("./FileFinder");

function connect(protocolVersion: string) {
  useJaato.setState((s) => ({ connection: { ...s.connection, phase: "connected", protocolVersion }, workspaceHidden: [] }));
}

async function type(value: string) {
  fireEvent.change(screen.getByRole("searchbox", { name: /Find a file/ }), { target: { value } });
  await act(async () => { await vi.advanceTimersByTimeAsync(300); });
}

beforeEach(() => { vi.useFakeTimers(); search.mockReset(); download.mockClear(); connect("1.32"); });
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe("FileFinder", () => {
  it("is not drawn against a daemon below 1.32", () => {
    connect("1.31");
    render(<FileFinder onView={() => undefined} />);
    expect(screen.queryByRole("searchbox")).toBeNull();
  });

  it("does not ask for a one-letter query", async () => {
    render(<FileFinder onView={() => undefined} />);
    await type("r");
    expect(search).not.toHaveBeenCalled();
  });

  it("finds a hidden file, marks it, and views it", async () => {
    useJaato.setState({ workspaceHidden: ["docs/"] });
    search.mockResolvedValue({ ok: true, total: 1, truncated: false, matches: [{ path: "docs/claimcascade-bundle.tgz", size: 2048, credential: false }] });
    const onView = vi.fn();
    render(<FileFinder onView={onView} />);
    await type("bundle");
    expect(search).toHaveBeenCalledWith("bundle");
    const row = screen.getByText("docs/claimcascade-bundle.tgz").closest("li")!;
    expect(row.textContent).toContain("H");
    fireEvent.click(screen.getByRole("button", { name: "View docs/claimcascade-bundle.tgz" }));
    expect(onView).toHaveBeenCalledWith("docs/claimcascade-bundle.tgz");
    fireEvent.click(screen.getByRole("button", { name: "Download docs/claimcascade-bundle.tgz" }));
    expect(download).toHaveBeenCalledWith("docs/claimcascade-bundle.tgz");
  });

  it("offers no view or download of a credential file", async () => {
    search.mockResolvedValue({ ok: true, total: 1, truncated: false, matches: [{ path: ".env", size: 10, credential: true }] });
    render(<FileFinder onView={() => undefined} />);
    await type(".env");
    expect(screen.getByText(".env")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /\.env/ })).toBeNull();
    expect(screen.getByText("credentials")).toBeTruthy();
  });

  it("says when the walk stopped early", async () => {
    search.mockResolvedValue({ ok: true, total: 0, truncated: true, matches: [] });
    render(<FileFinder onView={() => undefined} />);
    await type("zzz");
    expect(screen.getByText(/stopped early/)).toBeTruthy();
  });

  it("drops an answer to an older query", async () => {
    let resolveOld: (v: unknown) => void = () => undefined;
    search
      .mockImplementationOnce(() => new Promise((r) => { resolveOld = r; }))
      .mockResolvedValueOnce({ ok: true, total: 1, truncated: false, matches: [{ path: "new.txt", size: 1, credential: false }] });
    render(<FileFinder onView={() => undefined} />);
    await type("old");
    await type("new");
    await act(async () => { resolveOld({ ok: true, total: 1, truncated: false, matches: [{ path: "old.txt", size: 1, credential: false }] }); });
    expect(screen.getByText("new.txt")).toBeTruthy();
    expect(screen.queryByText("old.txt")).toBeNull();
  });
});
