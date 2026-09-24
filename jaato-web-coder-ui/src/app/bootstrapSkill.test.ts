/**
 * The jaato-sdk skill bootstrap (#1263).
 *
 * Two halves are pinned here:
 *
 *  1. the store REPORTS a ``scaffold.integration.result`` in the Files-panel
 *     notice — a SKIPPED refresh (an edited copy) and a real error surface,
 *     a clean apply is quiet on a re-check;
 *  2. the store subscription FIRES the daemon verb when a workspace or
 *     session context appears, once per context and again after a reconnect.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { JaatoEvent } from "@jaato/sdk";
import { useJaato } from "@/store/store";

const ev = (o: Record<string, unknown>) => o as unknown as JaatoEvent;

// ── connection module is mocked so the subscription can be observed without
// a real client; the bootstrap module self-installs its subscription at
// import, so it is imported AFTER the mock is registered. ──
const runScaffoldIntegration = vi.fn(async () => undefined);
let connected = true;
vi.mock("@/sdk/connection", () => ({
  isConnected: () => connected,
  getClient: () => ({ runScaffoldIntegration }),
}));

beforeEach(() => {
  useJaato.getState().resetSessionState();
  runScaffoldIntegration.mockClear();
  connected = true;
});

describe("scaffold.integration.result — the Files-panel notice", () => {
  it("reports a SKIPPED refresh, not silently", () => {
    useJaato.getState().dispatch([ev({
      type: "scaffold.integration.result", integration: "claude-code", ok: true,
      changed: false, state_before: "edited", state_after: "edited",
      skipped_reason: "the installed copy was changed since it was applied",
    })]);
    const notice = useJaato.getState().workspaceNotice!;
    expect(notice.error).toBeFalsy();
    expect(notice.text).toContain("left as-is");
    expect(notice.text).toContain("changed since it was applied");
  });

  it("reports an install once, and a re-check of a current copy is quiet", () => {
    useJaato.getState().dispatch([ev({
      type: "scaffold.integration.result", integration: "claude-code", ok: true,
      changed: true, state_before: "absent", state_after: "current",
      skipped_reason: "", server_version: "1.2.3",
    })]);
    expect(useJaato.getState().workspaceNotice!.text).toContain("installed");

    useJaato.getState().setWorkspaceNotice(null);
    useJaato.getState().dispatch([ev({
      type: "scaffold.integration.result", integration: "claude-code", ok: true,
      changed: false, state_before: "current", state_after: "current", skipped_reason: "",
    })]);
    expect(useJaato.getState().workspaceNotice).toBeNull();
  });

  it("reports a real failure as an error notice", () => {
    useJaato.getState().dispatch([ev({
      type: "scaffold.integration.result", integration: "claude-code", ok: false,
      error: "unknown integration 'claude-code'",
    })]);
    const notice = useJaato.getState().workspaceNotice!;
    expect(notice.error).toBe(true);
    expect(notice.text).toContain("unknown integration");
  });
});

describe("the store subscription fires the daemon verb", () => {
  it("fires once when a workspace appears and again for a new session context", async () => {
    await import("@/app/bootstrapSkill");
    const st = useJaato.getState();

    // The real store action that sets ``workspace.selected`` — what a
    // ``config.status`` reduces into after ``workspace.select``.
    st.selectWorkspace("project-a");
    await Promise.resolve();
    expect(runScaffoldIntegration).toHaveBeenCalledWith("claude-code");
    const afterWorkspace = runScaffoldIntegration.mock.calls.length;

    useJaato.setState(() => ({ sessionId: "20260101_000000" }));
    await Promise.resolve();
    expect(runScaffoldIntegration.mock.calls.length).toBeGreaterThan(afterWorkspace);
  });

  it("does not fire while disconnected", async () => {
    await import("@/app/bootstrapSkill");
    connected = false;
    runScaffoldIntegration.mockClear();
    useJaato.setState((s) => ({ workspace: { ...s.workspace, selected: "later" } }));
    await Promise.resolve();
    expect(runScaffoldIntegration).not.toHaveBeenCalled();
  });
});
