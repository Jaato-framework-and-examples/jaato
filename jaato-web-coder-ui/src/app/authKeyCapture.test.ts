import { describe, expect, it } from "vitest";
import { CAPTURE_WINDOW_MS, captureOnPostAuth, noteAuthKeyCommand, takePendingAuthKey } from "./authKeyCapture";
import type { CredentialsApi } from "./credentials";

const offer = (providerName: string) => ({ requestId: "r", providerName, providerDisplayName: providerName, models: [], hasActiveSession: false });

function recordingApi(): { api: CredentialsApi; added: Array<{ provider: string; secret: string }> } {
  const added: Array<{ provider: string; secret: string }> = [];
  const api: CredentialsApi = {
    async list() { return []; },
    async add(provider, secret) { added.push({ provider, secret }); return { id: "n", provider, label: provider, hint: "", createdAt: "" }; },
    async reveal() { return ""; },
    async remove() { /* noop */ },
  };
  return { api, added };
}

describe("auth key capture", () => {
  it("parks the secret of `<x>-auth key <secret>` and nothing else", () => {
    noteAuthKeyCommand("zhipuai-auth", ["login"]);
    expect(takePendingAuthKey()).toBeNull();
    noteAuthKeyCommand("theme", ["key", "x"]);
    expect(takePendingAuthKey()).toBeNull();
    noteAuthKeyCommand("zhipuai-auth", ["key"]);
    expect(takePendingAuthKey()).toBeNull();
    noteAuthKeyCommand("zhipuai-auth", ["KEY", "sk-1"]);
    expect(takePendingAuthKey()).toBe("sk-1");
    expect(takePendingAuthKey()).toBeNull(); // taken once
  });

  it("a parked secret older than the window is dropped unread", () => {
    noteAuthKeyCommand("nim-auth", ["key", "nvapi-1"], 1_000);
    expect(takePendingAuthKey(1_000 + CAPTURE_WINDOW_MS + 1)).toBeNull();
  });

  it("the daemon's offer files the secret under the provider IT names, not the command's prefix", async () => {
    const { api, added } = recordingApi();
    noteAuthKeyCommand("github-auth", ["key", "ghp_1"]);
    expect(await captureOnPostAuth(offer("github_models"), api)).toEqual({ provider: "github_models" });
    expect(added).toEqual([{ provider: "github_models", secret: "ghp_1" }]);
  });

  it("no parked secret, no store, or an offer without a provider stores nothing", async () => {
    const { api, added } = recordingApi();
    expect(await captureOnPostAuth(offer("zhipuai"), api)).toBeNull();
    noteAuthKeyCommand("zhipuai-auth", ["key", "sk-2"]);
    expect(await captureOnPostAuth(offer("zhipuai"), null)).toBeNull();
    noteAuthKeyCommand("zhipuai-auth", ["key", "sk-3"]);
    expect(await captureOnPostAuth(offer(""), api)).toBeNull();
    expect(await captureOnPostAuth(null, api)).toBeNull();
    expect(added).toEqual([]);
  });

  it("a store that refuses is swallowed: the sign-in already succeeded", async () => {
    const api: CredentialsApi = { async list() { return []; }, async add() { throw new Error("HTTP 500"); }, async reveal() { return ""; }, async remove() { /* noop */ } };
    noteAuthKeyCommand("zhipuai-auth", ["key", "sk-4"]);
    expect(await captureOnPostAuth(offer("zhipuai"), api)).toBeNull();
  });
});
