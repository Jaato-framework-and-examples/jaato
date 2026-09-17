import { beforeEach, describe, expect, it } from "vitest";
import { endDestination, exitOptions } from "./exitChoice";
import { useJaato } from "@/store/store";

beforeEach(() => {
  useJaato.getState().resetSessionState();
});

describe("exitOptions: the TUI's two option sets", () => {
  it("offers [d/e/r] to an idle session, in the TUI's order", () => {
    expect(exitOptions(false).map((o) => o.key)).toEqual(["d", "e", "r"]);
  });
  it("adds [c] first when a turn is running", () => {
    expect(exitOptions(true).map((o) => o.key)).toEqual(["c", "d", "e", "r"]);
    expect(exitOptions(true)[0]!.label).toBe("Cancel task and exit");
  });
  it("ends with Return either way, so Tab from the default reaches it last", () => {
    for (const running of [true, false]) expect(exitOptions(running).at(-1)!.key).toBe("r");
  });
});

describe("endDestination: where End session lands", () => {
  it("is the workspace list in workspace mode and the connect screen otherwise", () => {
    expect(endDestination("enabled")).toBe("workspaces");
    expect(endDestination("disabled")).toBe("connect");
    expect(endDestination("unknown")).toBe("connect");
  });
});

describe("the store's exit choice slice", () => {
  it("opens with focus on the first option, cycles focus, and closes", () => {
    const st = useJaato.getState();
    st.openExitChoice(false, exitOptions(false));
    expect(useJaato.getState().exitChoice).toMatchObject({ running: false, focus: 0 });
    useJaato.getState().focusExitChoice(2);
    expect(useJaato.getState().exitChoice?.focus).toBe(2);
    useJaato.getState().closeExitChoice();
    expect(useJaato.getState().exitChoice).toBeNull();
  });
  it("is dropped by the session state reset, so a deleted session leaves no open question", () => {
    useJaato.getState().openExitChoice(true, exitOptions(true));
    useJaato.getState().resetSessionState();
    expect(useJaato.getState().exitChoice).toBeNull();
  });
  it("focusing with no open choice is a no-op", () => {
    useJaato.getState().focusExitChoice(1);
    expect(useJaato.getState().exitChoice).toBeNull();
  });
});
