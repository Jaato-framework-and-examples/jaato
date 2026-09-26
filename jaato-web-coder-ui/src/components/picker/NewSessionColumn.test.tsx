/**
 * Start is disabled until a model is RESOLVED, and what it starts is what
 * the column showed.  Staged files stay client-side until Start.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("@/sdk/connection", () => ({ getClient: () => ({ fetchWorkspaceFile: vi.fn() }), isConnected: () => false }));
const attachFiles = vi.fn();
vi.mock("@/app/staging", () => ({ attachFiles: (...a: unknown[]) => attachFiles(...a) }));

import { NewSessionColumn, stageDrafts } from "./NewSessionColumn";
import { useJaato } from "@/store/store";

beforeEach(() => {
  attachFiles.mockReset();
  useJaato.setState({
    profiles: [
      { name: "validator", description: "checks", provider: "minimax", model: "MiniMax-M3" },
      { name: "analyst", description: "reads" },
    ],
    sessions: [],
  });
  useJaato.setState((s) => ({ workspace: { ...s.workspace, mode: "enabled", selected: undefined, config: undefined } }));
});
afterEach(() => cleanup());

const start = () => screen.getByRole("button", { name: /Start session/ });

describe("NewSessionColumn", () => {
  it("starts on default, with the other profiles collapsed and Start disabled until a model is picked", () => {
    const onStart = vi.fn();
    render(<NewSessionColumn onStart={onStart} />);
    expect(screen.getByTestId("base-profile")).toHaveTextContent("model: you select");
    expect(screen.queryByRole("list", { name: "Base profiles" })).toBeNull();
    expect(start()).toBeDisabled();
    expect(screen.getByTestId("start-summary")).toHaveTextContent("Select a model to start");
  });

  it("inherits a profile's model and starts with the profile only", () => {
    const onStart = vi.fn();
    render(<NewSessionColumn onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: "2 more base profiles" }));
    fireEvent.click(screen.getByRole("button", { name: "Use base profile validator" }));
    expect(screen.queryByRole("list", { name: "Base profiles" })).toBeNull(); // collapsed again
    expect(screen.getByRole("button", { name: "Inherit" })).toHaveAttribute("aria-pressed", "true");
    expect(start()).toBeEnabled();
    fireEvent.click(start());
    expect(onStart).toHaveBeenCalledWith("validator", undefined, []);
  });

  it("a profile with no model is enabled only once provider and model are both chosen", () => {
    const onStart = vi.fn();
    useJaato.setState((s) => ({ workspace: { ...s.workspace, selected: "w", config: { workspace: "w", configured: false, availableProviders: ["anthropic"], missingFields: [] } } }));
    render(<NewSessionColumn onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: "2 more base profiles" }));
    fireEvent.click(screen.getByRole("button", { name: "Use base profile analyst" }));
    expect(screen.getByText(/analyst defines none, select one/)).toHaveClass("text-warning");
    fireEvent.change(screen.getByRole("combobox", { name: "Provider" }), { target: { value: "anthropic" } });
    expect(start()).toBeDisabled();
    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), { target: { value: "claude-sonnet" } });
    expect(start()).toBeEnabled();
    fireEvent.click(start());
    expect(onStart).toHaveBeenCalledWith("analyst", { provider: "anthropic", model: "claude-sonnet" }, []);
  });

  it("stages nothing until Start, then each file under its own folder", () => {
    const onStart = vi.fn();
    render(<NewSessionColumn onStart={onStart} />);
    const file = new File(["x"], "spec.md", { type: "text/markdown" });
    fireEvent.change(screen.getByLabelText("Stage files"), { target: { files: [file] } });
    expect(screen.getByText("spec.md")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Folder for spec.md"), { target: { value: "docs" } });
    expect(attachFiles).not.toHaveBeenCalled();
    const drafts = [{ id: "d", file, name: "spec.md", size: 1, dir: "docs" }];
    stageDrafts(drafts);
    expect(attachFiles).toHaveBeenCalledWith([file], "docs");
  });
});
