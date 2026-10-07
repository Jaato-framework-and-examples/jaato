import { describe, expect, it } from "vitest";
import { SignInRequiredError, ticketProvider } from "./tickets";

function response(status: number, body?: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

describe("ticketProvider", () => {
  it("POSTs to the ticket URL with the session cookie and returns the ticket", async () => {
    const calls: Array<[string, RequestInit | undefined]> = [];
    const provider = ticketProvider({
      ticketUrl: "./api/ticket",
      loginUrl: "./auth/login",
      fetchImpl: async (url, init) => { calls.push([String(url), init]); return response(200, { ticket: "t-1" }); },
    });
    expect(await provider()).toBe("t-1");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.[0]).toBe("./api/ticket");
    expect(calls[0]?.[1]?.method).toBe("POST");
    expect(calls[0]?.[1]?.credentials).toBe("same-origin");
  });

  it("asks the backend again on every call: a ticket is single-use", async () => {
    let n = 0;
    const provider = ticketProvider({ ticketUrl: "./api/ticket", loginUrl: "./auth/login", fetchImpl: async () => response(200, { ticket: `t-${++n}` }) });
    expect(await provider()).toBe("t-1");
    expect(await provider()).toBe("t-2");
  });

  it("turns a 401 into SignInRequiredError carrying the login URL", async () => {
    const provider = ticketProvider({ ticketUrl: "./api/ticket", loginUrl: "./auth/login", fetchImpl: async () => response(401) });
    await expect(provider()).rejects.toBeInstanceOf(SignInRequiredError);
    await expect(provider()).rejects.toMatchObject({ loginUrl: "./auth/login" });
  });

  it("reports other failures as errors, never as a credential", async () => {
    await expect(ticketProvider({ ticketUrl: "u", loginUrl: "l", fetchImpl: async () => response(503) })()).rejects.toThrow(/capacity|unreachable/);
    await expect(ticketProvider({ ticketUrl: "u", loginUrl: "l", fetchImpl: async () => response(500) })()).rejects.toThrow(/HTTP 500/);
    await expect(ticketProvider({ ticketUrl: "u", loginUrl: "l", fetchImpl: async () => response(200, {}) })()).rejects.toThrow(/no ticket/);
  });
});
