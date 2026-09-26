/**
 * The delete panel asks for the typed name whenever anything would be
 * lost -- and whenever it could not find out -- and is one click only for a
 * workspace demonstrably empty.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

const inspectWorkspace = vi.fn();
vi.mock("@/sdk/connection", () => ({ inspectWorkspace: (n: string) => inspectWorkspace(n) }));

import { DeletePanel } from "./DeletePanel";

afterEach(() => { cleanup(); inspectWorkspace.mockReset(); });

describe("DeletePanel", () => {
  it("shows the impact and enables Delete only on an exact typed match", async () => {
    inspectWorkspace.mockResolvedValue({
      name: "fixes", ok: true, path: "/root/.jaato/workspaces/fixes", size_bytes: 412 * 1024 * 1024,
      sessions: { total: 5, waiting: 2, awake: 1, sleeping: 2 },
      repos: [{ forge: "github", repo: "jaato-framework/jaato", branch: "main", path: "jaato", uncommitted: 5, unpushed: 1 }],
    });
    const onDelete = vi.fn();
    render(<DeletePanel name="fixes" busy={false} onDelete={onDelete} onCancel={() => undefined} />);
    expect(await screen.findByText("5 sessions deleted")).toBeInTheDocument();
    expect(screen.getByText("Open prompts are cancelled. Running work stops.")).toBeInTheDocument();
    expect(screen.getByText("⚠ 5 uncommitted files, 1 commit not pushed, lost")).toBeInTheDocument();
    expect(screen.getByText("412 MB")).toBeInTheDocument();
    const del = screen.getByRole("button", { name: "Delete workspace fixes permanently" });
    expect(del).toBeDisabled();
    const input = screen.getByRole("textbox", { name: "Type fixes to confirm" });
    fireEvent.change(input, { target: { value: "fixe" } });
    expect(input).toHaveClass("input-warn");
    expect(del).toBeDisabled();
    fireEvent.change(input, { target: { value: " fixes " } });
    expect(del).toBeEnabled();
    fireEvent.click(del);
    expect(onDelete).toHaveBeenCalled();
  });

  it("an empty workspace is one click", async () => {
    inspectWorkspace.mockResolvedValue({ name: "e", ok: true, sessions: { total: 0 }, repos: [], size_bytes: 0 });
    render(<DeletePanel name="e" busy={false} onDelete={() => undefined} onCancel={() => undefined} />);
    expect(await screen.findByText("Nothing to lose: no sessions, no repositories, no staged files.")).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.getByRole("button", { name: "Delete workspace e permanently" })).toBeEnabled();
  });

  it("a daemon that could not answer still asks for the name", async () => {
    inspectWorkspace.mockResolvedValue(null);
    render(<DeletePanel name="old" busy={false} onDelete={() => undefined} onCancel={() => undefined} />);
    expect(await screen.findByText(/Could not check what this workspace holds/)).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Type old to confirm" })).toBeInTheDocument();
  });
});
