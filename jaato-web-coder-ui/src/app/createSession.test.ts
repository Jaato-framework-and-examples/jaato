/**
 * ``createSession`` binds the screen to a new session, so the screen
 * starts empty — the property ``attachSession`` has always had and this
 * verb did not, which is how a failed attempt's errors ended up at the
 * top of the next session's transcript.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const sdkCreateSession = vi.fn(async () => undefined);
vi.mock("@/sdk/connection", () => ({
  getClient: () => ({ createSession: sdkCreateSession }),
  disconnect: vi.fn(async () => undefined),
  isConnected: () => true,
  reassertAfterReconnect: vi.fn(async () => undefined),
  selectWorkspace: vi.fn(async () => undefined),
}));

import { createSession } from "./actions";
import { MAIN_AGENT, useJaato } from "@/store/store";

beforeEach(() => {
  sdkCreateSession.mockClear();
  useJaato.getState().resetSessionState();
  useJaato.setState({ uploads: [] });
});

describe("createSession", () => {
  it("empties the pane before the session is asked for", async () => {
    const st = useJaato.getState();
    st.addSystemBlock(MAIN_AGENT, "[RunnerBootstrapFailed] the previous attempt", "error");
    expect(useJaato.getState().blocks[MAIN_AGENT]).toHaveLength(1);
    await createSession(null);
    expect(useJaato.getState().blocks[MAIN_AGENT]).toEqual([]);
    expect(sdkCreateSession).toHaveBeenCalledWith({});
  });

  it("keeps the files waiting to be staged, which the picker opens the session for", async () => {
    useJaato.getState().addUploads([{ id: "u1", path: "brief.txt", size: 3, status: "queued" }]);
    await createSession("researcher");
    expect(useJaato.getState().uploads.map((u) => u.id)).toEqual(["u1"]);
    expect(sdkCreateSession).toHaveBeenCalledWith({ profile: "researcher" });
  });

  it("drops the previous session's id, so nothing is answered against it", async () => {
    useJaato.setState({ sessionId: "20260917_085132" });
    await createSession(null);
    expect(useJaato.getState().sessionId).toBeNull();
  });
});
