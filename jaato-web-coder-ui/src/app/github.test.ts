import { describe, expect, it } from "vitest";
import { describeAccount, githubApi, type GitHubAccount } from "./github";
import { SignInRequiredError } from "./tickets";

function response(status: number, body: unknown = {}): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

const ACC = (o: Partial<GitHubAccount> & { id: string; login: string }): GitHubAccount => ({
  name: null, noreplyEmail: "1+x@users.noreply.github.com", installations: [], isDefault: false,
  createdAt: "2026-09-01T00:00:00Z", updatedAt: "2026-09-01T00:00:00Z", ...o,
});

describe("github client", () => {
  it("lists accounts under /accounts with the cookie, and never a token", async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const api = githubApi("./api/github", async (u, init) => {
      calls.push({ url: String(u), init: init ?? {} });
      return response(200, { accounts: [ACC({ id: "a", login: "alice", isDefault: true }), null] });
    });
    const accounts = await api.listAccounts();
    expect(accounts.map((a) => a.login)).toEqual(["alice"]);
    // No field named anything token-like is even declared on the type — the
    // whole point of the feature — and none is read here.
    expect(JSON.stringify(accounts[0])).not.toMatch(/token|secret/i);
    expect(calls[0]!.url).toBe("./api/github/accounts");
    expect(calls[0]!.init.credentials).toBe("same-origin");
  });

  it("has no reveal: there is no way to fetch a secret value", () => {
    const api = githubApi("./api/github", async () => response(200));
    expect((api as unknown as Record<string, unknown>).reveal).toBeUndefined();
    expect(Object.keys(api).sort()).toEqual(["bind", "disconnect", "listAccounts", "listBindings", "setDefault"]);
  });

  it("lists bindings under /bindings and filters malformed rows", async () => {
    const api = githubApi("/app/api/github", async (u) => {
      expect(String(u)).toBe("/app/api/github/bindings");
      return response(200, { bindings: [{ workspace: "ws1", accountId: "a" }, { workspace: 3 }, null] });
    });
    expect(await api.listBindings()).toEqual([{ workspace: "ws1", accountId: "a" }]);
  });

  it("setDefault posts {id} to /default and returns the accounts", async () => {
    const calls: Array<{ url: string; method?: string; body?: string }> = [];
    const api = githubApi("./api/github", async (u, init) => {
      calls.push({ url: String(u), method: init?.method, body: init?.body as string });
      return response(200, { accounts: [ACC({ id: "b", login: "bob", isDefault: true })] });
    });
    expect((await api.setDefault("b"))[0]!.login).toBe("bob");
    expect(calls[0]).toMatchObject({ url: "./api/github/default", method: "POST" });
    expect(JSON.parse(calls[0]!.body!)).toEqual({ id: "b" });
  });

  it("disconnect posts {id} to /disconnect and returns the login dropped", async () => {
    const calls: Array<{ url: string; body?: string }> = [];
    const api = githubApi("./api/github", async (u, init) => {
      calls.push({ url: String(u), body: init?.body as string });
      return response(200, { disconnected: "alice", accounts: [] });
    });
    const r = await api.disconnect("a");
    expect(r).toEqual({ disconnected: "alice", accounts: [] });
    expect(calls[0]!.url).toBe("./api/github/disconnect");
    expect(JSON.parse(calls[0]!.body!)).toEqual({ id: "a" });
  });

  it("bind posts {workspace, account_id} and normalises the result; null clears", async () => {
    const calls: Array<{ body?: string }> = [];
    const api = githubApi("./api/github", async (u, init) => {
      expect(String(u)).toBe("./api/github/bind");
      calls.push({ body: init?.body as string });
      return response(200, { binding: "set", envWritten: true, gitconfigSeeded: true, reloaded: 2 });
    });
    const set = await api.bind("ws1", "a");
    expect(set).toEqual({ binding: "set", envWritten: true, gitconfigSeeded: true, reloaded: 2, note: undefined });
    expect(JSON.parse(calls[0]!.body!)).toEqual({ workspace: "ws1", account_id: "a" });
    await api.bind("ws1", null);
    expect(JSON.parse(calls[1]!.body!)).toEqual({ workspace: "ws1", account_id: null });
  });

  it("a 401 is the sign-in error; other failures carry the backend's reason", async () => {
    await expect(githubApi("u", async () => response(401)).listAccounts()).rejects.toBeInstanceOf(SignInRequiredError);
    await expect(githubApi("u", async () => response(404, { error: "no such connected account" })).setDefault("x")).rejects.toThrow(/HTTP 404 \(no such connected account\)/);
  });

  it("describeAccount marks the default", () => {
    expect(describeAccount(ACC({ id: "a", login: "alice", isDefault: true }))).toBe("@alice (default)");
    expect(describeAccount(ACC({ id: "b", login: "bob" }))).toBe("@bob");
  });
});
