import { describe, expect, it } from "vitest";
import { answerOfferDownload, refusalText, servesDownloads } from "./downloads";

const client = (event: Record<string, unknown>) => ({
  fetchWorkspaceFile: async (_path: string, opts?: { metadataOnly?: boolean }) => {
    expect(opts?.metadataOnly).toBe(true); // the tool checks; the bytes move on click
    return { event: event as never, data: null };
  },
});

describe("offer_download", () => {
  it("answers the model with what it offered", async () => {
    const out = await answerOfferDownload({ path: "out/r.pdf" }, client({ ok: true, path: "out/r.pdf", name: "r.pdf", size: 3, mime_type: "application/pdf" }));
    expect(out).toMatchObject({ offered: true, path: "out/r.pdf", size: 3 });
  });
  it("throws the daemon's refusal so the model is told why, and nothing is offered", async () => {
    await expect(answerOfferDownload({ path: ".env" }, client({ ok: false, category: "credential" }))).rejects.toThrow(".env: holds credentials");
  });
  it("refuses a call with no path before asking the daemon", async () => {
    await expect(answerOfferDownload({}, { fetchWorkspaceFile: async () => { throw new Error("asked"); } })).rejects.toThrow("needs a path");
  });
});

describe("download plumbing", () => {
  it("is offered only by a daemon that serves the fetch", () => {
    expect(servesDownloads("1.20")).toBe(true);
    expect(servesDownloads("1.19")).toBe(false);
    expect(servesDownloads(null)).toBe(false);
  });
  it("names a refusal in the reader's words, and keeps a category it does not know", () => {
    expect(refusalText("too_large", "x")).toBe("too large to download");
    expect(refusalText("brand_new", "daemon says so")).toBe("daemon says so");
  });
});
