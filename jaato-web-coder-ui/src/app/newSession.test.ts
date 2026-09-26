/**
 * The New session column's three model cases, decided by the base profile
 * alone, and what Start sends for each.  The property worth pinning is the
 * one a wrong answer costs most: an INHERITED model sends no override, so
 * the daemon resolves the profile's own binding rather than a copy the
 * picker read earlier.
 */
import { describe, expect, it } from "vitest";
import { baseProfiles, modelStatus, modelTag, resolveModel, startRequest, startSummary, knownModels } from "./newSession";

const profiles = [
  { name: "validator", description: "checks", provider: "minimax", model: "MiniMax-M3" },
  { name: "analyst", description: "reads" },
];

describe("baseProfiles", () => {
  it("puts a synthesised default first and sorts the rest", () => {
    const b = baseProfiles(profiles);
    expect(b.map((p) => p.name)).toEqual(["default", "analyst", "validator"]);
    expect(b[0]).toMatchObject({ isDefault: true, model: "" });
  });
  it("uses a daemon-listed default profile instead of the synthesised one", () => {
    const b = baseProfiles([{ name: "default", provider: "anthropic", model: "claude" }]);
    expect(b).toHaveLength(1);
    expect(modelTag(b[0]!)).toBe("model: anthropic / claude");
  });
});

describe("model cases", () => {
  const [def, analyst, validator] = baseProfiles(profiles);
  const none = { provider: "", model: "" };
  const pick = { provider: "anthropic", model: "claude-sonnet" };

  it("default requires a pick and sends no --profile", () => {
    expect(modelTag(def!)).toBe("model: you select");
    expect(resolveModel(def!, "inherit", none)).toBeNull();
    expect(startRequest(def!, "inherit", none)).toBeNull();
    expect(modelStatus(def!, "inherit", none)).toEqual({ text: "selected", warn: false });
    expect(startRequest(def!, "inherit", pick)).toEqual({ profile: null, model: pick });
  });

  it("a profile with a model inherits it by default and sends only the profile", () => {
    expect(startRequest(validator!, "inherit", none)).toEqual({ profile: "validator" });
    expect(modelStatus(validator!, "inherit", none).text).toBe("inherited from validator");
    expect(startSummary(validator!, resolveModel(validator!, "inherit", none), 2)).toBe("validator · minimax / MiniMax-M3 · 2 files");
  });

  it("overriding needs a pick and sends it with the profile", () => {
    expect(startRequest(validator!, "override", none)).toBeNull();
    expect(modelStatus(validator!, "override", none).text).toBe("overriding MiniMax-M3");
    expect(startRequest(validator!, "override", pick)).toEqual({ profile: "validator", model: pick });
  });

  it("a profile with no model warns until one is picked", () => {
    expect(modelTag(analyst!)).toBe("model: not defined");
    expect(modelStatus(analyst!, "inherit", none)).toEqual({ text: "analyst defines none, select one", warn: true });
    expect(modelStatus(analyst!, "inherit", pick).warn).toBe(false);
    expect(startRequest(analyst!, "inherit", pick)).toEqual({ profile: "analyst", model: pick });
    expect(startSummary(analyst!, null, 0)).toBe("Select a model to start");
  });

  it("a provider without a model is not a model", () => {
    expect(resolveModel(def!, "inherit", { provider: "anthropic", model: "  " })).toBeNull();
  });
});

describe("knownModels", () => {
  it("suggests only models seen with the chosen provider", () => {
    const seen = [{ provider: "a", model: "m1" }, { provider: "b", model: "m2" }, { provider: "a", model: "m1" }];
    expect(knownModels("a", seen)).toEqual(["m1"]);
  });
});

describe("default on a daemon with no workspace mode", () => {
  const [def] = baseProfiles([], { envFallback: true });
  it("starts on the daemon's .env with nothing picked, and overrides once a model is picked", () => {
    expect(startRequest(def!, "inherit", { provider: "", model: "" })).toEqual({ profile: null });
    expect(startSummary(def!, resolveModel(def!, "inherit", { provider: "", model: "" }), 0)).toBe("default · daemon .env");
    expect(startRequest(def!, "inherit", { provider: "a", model: "m" })).toEqual({ profile: null, model: { provider: "a", model: "m" } });
  });
});
