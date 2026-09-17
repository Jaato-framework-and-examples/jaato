import { describe, expect, it } from "vitest";
import { fetchSignedInUser, siblingEndpoint } from "./backendSession";

function response(status: number, body: string, type = "application/json; charset=utf-8"): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": type }),
    json: async () => JSON.parse(body),
  } as unknown as Response;
}

describe("siblingEndpoint", () => {
  it("keeps the ticket URL's directory and swaps the last segment", () => {
    expect(siblingEndpoint("./api/ticket", "session")).toBe("./api/session");
    expect(siblingEndpoint("/api/ticket", "logout")).toBe("/api/logout");
    expect(siblingEndpoint("https://h/app/api/ticket?x=1", "session")).toBe("https://h/app/api/session");
    expect(siblingEndpoint("ticket", "session")).toBe("session");
  });
});

describe("fetchSignedInUser", () => {
  it("returns the user the backend reports", async () => {
    const calls: RequestInit[] = [];
    const who = await fetchSignedInUser("./api/session", async (_u, init) => { calls.push(init ?? {}); return response(200, '{"user":"alice","expiresAt":"2026-09-16T10:00:00Z"}'); });
    expect(who).toEqual({ user: "alice" });
    expect(calls[0]?.credentials).toBe("same-origin");
  });

  it("reads a 401 as nobody signed in", async () => {
    expect(await fetchSignedInUser("./api/session", async () => response(401, '{"error":"no session"}'))).toBeNull();
  });

  it("reads the dev server's HTML fallback and a network failure as unknown, never as an error", async () => {
    expect(await fetchSignedInUser("./api/session", async () => response(200, "{}", "text/html"))).toBeNull();
    expect(await fetchSignedInUser("./api/session", async () => { throw new TypeError("offline"); })).toBeNull();
    expect(await fetchSignedInUser("./api/session", async () => response(200, '{"user":""}'))).toBeNull();
  });
});
