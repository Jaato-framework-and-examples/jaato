import { describe, expect, it } from "vitest";
import { loadLauncherConfig, parseLauncherConfig } from "./launcherConfig";

function response(body: string, { ok = true, type = "application/json; charset=utf-8" } = {}): Response {
  return {
    ok,
    headers: new Headers({ "content-type": type }),
    json: async () => JSON.parse(body),
  } as unknown as Response;
}

describe("parseLauncherConfig", () => {
  it("keeps only the known keys with the right types", () => {
    expect(parseLauncherConfig({ daemon: "ws://d:1", token: "t", autoConnect: true, extra: 1 })).toEqual({ daemon: "ws://d:1", token: "t", autoConnect: true });
    expect(parseLauncherConfig({ daemon: "wss://d", ticketUrl: "./api/ticket", loginUrl: "./auth/login" })).toEqual({ daemon: "wss://d", ticketUrl: "./api/ticket", loginUrl: "./auth/login" });
    expect(parseLauncherConfig({ ticketUrl: "./api/ticket", sessionUrl: "./who", logoutUrl: "./bye" })).toEqual({ ticketUrl: "./api/ticket", sessionUrl: "./who", logoutUrl: "./bye" });
    expect(parseLauncherConfig({ daemon: 5, token: "", autoConnect: "yes" })).toEqual({});
    expect(parseLauncherConfig(null)).toEqual({});
    expect(parseLauncherConfig("x")).toEqual({});
  });
});

describe("loadLauncherConfig", () => {
  it("returns the launcher's config", async () => {
    const cfg = await loadLauncherConfig(async () => response('{"daemon":"ws://127.0.0.1:8080","token":"abc","autoConnect":true}'));
    expect(cfg).toEqual({ daemon: "ws://127.0.0.1:8080", token: "abc", autoConnect: true });
  });

  it("treats the dev server's SPA fallback (HTML at /config.json) as no config", async () => {
    const cfg = await loadLauncherConfig(async () => response("<!doctype html>", { type: "text/html" }));
    expect(cfg).toEqual({});
  });

  it("treats 404 and network failure as no config", async () => {
    expect(await loadLauncherConfig(async () => response("{}", { ok: false }))).toEqual({});
    expect(await loadLauncherConfig(async () => { throw new Error("offline"); })).toEqual({});
  });
});
