import { describe, expect, it } from "vitest";
import { applyCloneProgress, formatBytes, formatSource, needsTypedConfirm, normalizeInspection, normalizeSources, repoLoss, validWorkspaceName, type CloneRow } from "./workspaces";

describe("sources", () => {
  it("renders forge: owner/repo@branch, and a checkout with no remote by its path", () => {
    const s = normalizeSources([
      { forge: "github", repo: "acme/claims", branch: "main", path: "claims" },
      { forge: "", repo: "", branch: "dev", path: "scratch" },
      "junk",
    ]);
    expect(s.map(formatSource)).toEqual(["github: acme/claims@main", "scratch@dev (no remote)"]);
    expect(normalizeSources(undefined)).toEqual([]);
  });
});

describe("inspection", () => {
  const empty = normalizeInspection({ name: "w", ok: true, sessions: { total: 0 }, repos: [], size_bytes: 10 });
  it("asks for the typed name when there is anything to lose, or when it could not look", () => {
    expect(needsTypedConfirm(empty)).toBe(false);
    expect(needsTypedConfirm(normalizeInspection({ ok: true, sessions: { total: 2, sleeping: 2 } }))).toBe(true);
    expect(needsTypedConfirm(normalizeInspection({ ok: true, repos: [{ repo: "a/b", path: "b" }] }))).toBe(true);
    expect(needsTypedConfirm(normalizeInspection({ ok: false, error: "nope" }))).toBe(true);
    expect(needsTypedConfirm(null)).toBe(true);
  });
  it("says what a repository loses", () => {
    const i = normalizeInspection({ repos: [{ repo: "a/b", path: "b", uncommitted: 5, unpushed: 1 }, { repo: "c/d", path: "d", uncommitted: 0, unpushed: null }] });
    expect(i.repos.map(repoLoss)).toEqual(["5 uncommitted files, 1 commit not pushed", ""]);
  });
  it("formats sizes", () => {
    expect(formatBytes(null)).toBe("size unknown");
    expect(formatBytes(412 * 1024 * 1024)).toBe("412 MB");
  });
});

describe("applyCloneProgress", () => {
  const rows: CloneRow[] = [
    { repo: "a/b", branch: "main", state: "queued", percent: 0, error: "" },
    { repo: "c/d", branch: "dev", state: "queued", percent: 0, error: "" },
  ];
  it("updates the row the event names, and keeps a failed bar where it stopped", () => {
    let r = applyCloneProgress(rows, { repo: "a/b", state: "cloning", percent: 56 });
    expect(r[0]).toMatchObject({ state: "cloning", percent: 56 });
    r = applyCloneProgress(r, { repo: "a/b", state: "failed", error: "auth" });
    expect(r[0]).toMatchObject({ state: "failed", percent: 56, error: "auth" });
    expect(r[1]!.state).toBe("queued");
  });
  it("a done row is at 100%", () => {
    expect(applyCloneProgress(rows, { repo: "c/d", state: "done" })[1]).toMatchObject({ state: "done", percent: 100 });
  });
  it("an event naming no repository fails every row not already done", () => {
    const r = applyCloneProgress([{ ...rows[0]!, state: "done", percent: 100 }, rows[1]!], { repo: "", state: "failed", error: "not visible" });
    expect(r.map((x) => x.state)).toEqual(["done", "failed"]);
  });
  it("ignores an unknown state", () => {
    expect(applyCloneProgress(rows, { repo: "a/b", state: "exploded" })).toBe(rows);
  });
});

describe("validWorkspaceName", () => {
  it("takes one flat component", () => {
    expect(validWorkspaceName("claims-refactor_2.x")).toBe(true);
    for (const bad of ["", "a/b", "..", ".", "has space"]) expect(validWorkspaceName(bad)).toBe(false);
  });
});
