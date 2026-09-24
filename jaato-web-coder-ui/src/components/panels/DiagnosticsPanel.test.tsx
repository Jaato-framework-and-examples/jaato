/**
 * The Diagnostics panel (#1294): the record and the live re-check are
 * shown in two visually separate blocks, the "Re-check now" action is the
 * ONLY action, and there is no export / download / clipboard-copy control
 * anywhere on this panel.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useJaato, emptyDiagnostics } from "@/store/store";
import type { DiagnosticsState } from "@/store/types";

const refreshDiagnostics = vi.fn(async () => undefined);
vi.mock("@/app/diagnostics", async (orig) => ({
  ...(await orig<typeof import("@/app/diagnostics")>()),
  refreshDiagnostics: () => refreshDiagnostics(),
}));

const { DiagnosticsPanel } = await import("./DiagnosticsPanel");

function load(patch: Partial<DiagnosticsState>) {
  useJaato.setState({ sessionId: "s1", diagnostics: { ...emptyDiagnostics(), status: "loaded", checkedAt: Date.now(), ...patch } });
}

beforeEach(() => { refreshDiagnostics.mockClear(); });
afterEach(() => { cleanup(); });

describe("DiagnosticsPanel", () => {
  it("shows the record fields and the live probe in two distinct sections", () => {
    load({
      confinementId: "jaato-ws-mine-abc",
      sandboxMode: "apparmor",
      probe: { ok: true, error: "", expected_profile: "jaato-ws-mine-abc", current_profile: "jaato-ws-mine-abc", current_mode: "enforce", enforced: true, confined: true, scan: null },
    });
    render(<DiagnosticsPanel />);
    expect(screen.getByRole("region", { name: "Session record" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Live confinement check" })).toBeTruthy();
    expect(screen.getByText("Enforced")).toBeTruthy();
  });

  it("a cached record and a disagreeing live probe are both visible, unmerged (#1253)", () => {
    load({
      sandboxMode: "apparmor",
      probe: { ok: true, error: "", expected_profile: "x", current_profile: "unconfined", current_mode: null, enforced: false, confined: false, scan: null },
    });
    render(<DiagnosticsPanel />);
    const record = screen.getByRole("region", { name: "Session record" });
    const live = screen.getByRole("region", { name: "Live confinement check" });
    expect(record.textContent).toContain("apparmor");
    expect(live.textContent).toContain("Not confined");
  });

  it("a probe that could not determine an answer says so, never a guessed verdict", () => {
    load({ probe: { ok: false, error: "thread scan failed", expected_profile: "x", current_profile: "", current_mode: null, enforced: false, confined: false, scan: null } });
    render(<DiagnosticsPanel />);
    expect(screen.getByText(/Could not determine confinement/)).toBeTruthy();
    expect(screen.queryByText("Enforced")).toBeNull();
    expect(screen.queryByText("Not confined")).toBeNull();
  });

  it("Re-check now calls the daemon verb again", () => {
    load({ probe: null });
    render(<DiagnosticsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Re-check now" }));
    expect(refreshDiagnostics).toHaveBeenCalledTimes(1);
  });

  it("a refusal is shown as an alert, in words", () => {
    useJaato.setState({ sessionId: "s1", diagnostics: { ...emptyDiagnostics(), status: "error", error: "Only the owner of this workspace can view its diagnostics." } });
    render(<DiagnosticsPanel />);
    expect(screen.getByRole("alert").textContent).toContain("Only the owner");
  });

  it("offers no export, download or copy control", () => {
    load({
      probe: { ok: true, error: "", expected_profile: "x", current_profile: "x", current_mode: "enforce", enforced: true, confined: true, scan: null },
    });
    render(<DiagnosticsPanel />);
    for (const word of [/export/i, /download/i, /copy/i]) {
      expect(screen.queryByRole("button", { name: word })).toBeNull();
    }
  });

  it("nothing checked yet says so instead of rendering a blank verdict", () => {
    useJaato.setState({ sessionId: "s1", diagnostics: emptyDiagnostics() });
    render(<DiagnosticsPanel />);
    expect(screen.getByText("never checked")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Live confinement check" })).toBeNull();
  });

  it("a daemon below 1.25 is named, not silently empty", () => {
    useJaato.setState({ sessionId: "s1", diagnostics: { ...emptyDiagnostics(), status: "unsupported" } });
    render(<DiagnosticsPanel />);
    expect(screen.getByText(/older than protocol 1.25/)).toBeTruthy();
  });
});
