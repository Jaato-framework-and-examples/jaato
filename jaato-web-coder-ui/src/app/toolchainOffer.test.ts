/**
 * The page stages ``.jaato/toolchain-offer.json`` (the backend may not be
 * able to write the workspace).  Pinned here:
 *   • the offer is staged through ``stageFiles`` under the backend's path,
 *     with the backend's content byte for byte;
 *   • only into this connection's workspace, never a status for another;
 *   • the same content is staged once, a changed one again;
 *   • a status with no offer (an older backend) stages nothing;
 *   • a staging the daemon refused is not remembered, so the next status retries.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useJaato } from "@/store/store";

const stageFiles = vi.fn();
let connected = true;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => connected,
  getClient: () => ({ stageFiles }),
  reassertAfterReconnect: async () => undefined,
}));

const { stageToolchainOffer, resetStagedOffers } = await import("@/app/toolchainOffer");

const WS = "/srv/ws/alice-1";
const offer = (content: string) => ({ workspace: WS, offer: { path: ".jaato/toolchain-offer.json", content } });
const sentText = (call: unknown[]) => new TextDecoder().decode((call[1] as Array<{ data: Uint8Array }>)[0]!.data);

beforeEach(() => {
  connected = false;
  useJaato.getState().resetSessionState();
  useJaato.setState((s) => ({ workspace: { ...s.workspace, selected: "alice-1", list: [{ name: "alice-1", path: WS } as never] } }));
  stageFiles.mockReset();
  stageFiles.mockImplementation(async (_ws: string, files: Array<{ name: string }>) => ({ staged: files.map((f) => f.name), failed: [] }));
  resetStagedOffers();
  connected = true;
});

describe("staging the toolchain offer", () => {
  it("stages the backend's content under its path into the connection's workspace", async () => {
    expect(await stageToolchainOffer(offer('{"schema":1,"toolchains":[]}'))).toBe(true);
    expect(stageFiles).toHaveBeenCalledTimes(1);
    expect(stageFiles.mock.calls[0]![0]).toBe("");
    expect((stageFiles.mock.calls[0]![1] as Array<{ name: string }>)[0]!.name).toBe(".jaato/toolchain-offer.json");
    expect(sentText(stageFiles.mock.calls[0]!)).toBe('{"schema":1,"toolchains":[]}');
  });

  it("never stages a status for another workspace", async () => {
    expect(await stageToolchainOffer({ ...offer("{}"), workspace: "/srv/ws/other" })).toBe(false);
    expect(stageFiles).not.toHaveBeenCalled();
  });

  it("stages the same content once, and a changed one again", async () => {
    await stageToolchainOffer(offer("a"));
    await stageToolchainOffer(offer("a"));
    await stageToolchainOffer(offer("b"));
    expect(stageFiles.mock.calls.map(sentText)).toEqual(["a", "b"]);
  });

  it("an older backend's status (no offer) stages nothing", async () => {
    expect(await stageToolchainOffer({ workspace: WS })).toBe(false);
    expect(stageFiles).not.toHaveBeenCalled();
  });

  it("a refused staging is retried on the next status", async () => {
    stageFiles.mockResolvedValueOnce({ staged: [], failed: [{ name: ".jaato/toolchain-offer.json", category: "io_error" }] });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(await stageToolchainOffer(offer("a"))).toBe(false);
    expect(await stageToolchainOffer(offer("a"))).toBe(true);
    expect(stageFiles).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});
