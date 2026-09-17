import { describe, expect, it } from "vitest";
import { credentialsApi, describeCredential } from "./credentials";
import { SignInRequiredError } from "./tickets";

function response(status: number, body: unknown = {}): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("credentials client", () => {
  it("lists per provider with the cookie, and reads labels and hints, never a secret", async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const api = credentialsApi("./api/credentials", async (u, init) => {
      calls.push({ url: String(u), init: init ?? {} });
      return response(200, { entries: [{ id: "a", provider: "zhipuai", label: "work", hint: "mnop", createdAt: "2026-09-16T10:00:00Z" }, null] });
    });
    const list = await api.list("zhipuai");
    expect(list.map((e) => e.id)).toEqual(["a"]);
    expect(calls[0]!.url).toBe("./api/credentials?provider=zhipuai");
    expect(calls[0]!.init.credentials).toBe("same-origin");
  });

  it("add posts JSON and returns the entry; reveal posts to the entry's reveal route; remove tolerates 404", async () => {
    const calls: Array<{ url: string; method?: string; body?: string }> = [];
    const api = credentialsApi("/app/api/credentials", async (u, init) => {
      calls.push({ url: String(u), method: init?.method, body: init?.body as string | undefined });
      if (String(u).endsWith("/reveal")) return response(200, { secret: "sk-1" });
      if (init?.method === "DELETE") return response(404, { error: "no such credential" });
      return response(201, { entry: { id: "x", provider: "zhipuai", label: "zhipuai …mnop", hint: "mnop", createdAt: "t" } });
    });
    expect((await api.add("zhipuai", "sk-abcdefghijklmnop")).id).toBe("x");
    expect(JSON.parse(calls[0]!.body!)).toEqual({ provider: "zhipuai", secret: "sk-abcdefghijklmnop" });
    expect(await api.reveal("x y")).toBe("sk-1");
    expect(calls[1]).toMatchObject({ url: "/app/api/credentials/x%20y/reveal", method: "POST" });
    await api.remove("x");
    expect(calls[2]).toMatchObject({ url: "/app/api/credentials/x", method: "DELETE" });
  });

  it("a 401 is the sign-in error; other failures carry the backend's reason", async () => {
    await expect(credentialsApi("u", async () => response(401)).list("zhipuai")).rejects.toBeInstanceOf(SignInRequiredError);
    await expect(credentialsApi("u", async () => response(400, { error: "provider must be…" })).add("Bad", "x".repeat(20))).rejects.toThrow(/HTTP 400 \(provider must be…\)/);
    await expect(credentialsApi("u", async () => response(200, {})).reveal("x")).rejects.toThrow(/no secret/);
  });

  it("describeCredential adds the hint unless the label already ends in it", () => {
    expect(describeCredential({ id: "a", provider: "p", label: "zhipuai …mnop", hint: "mnop", createdAt: "" })).toBe("zhipuai …mnop");
    expect(describeCredential({ id: "a", provider: "p", label: "work", hint: "mnop", createdAt: "" })).toBe("work (…mnop)");
    expect(describeCredential({ id: "a", provider: "p", label: "work", hint: "", createdAt: "" })).toBe("work");
  });
});
