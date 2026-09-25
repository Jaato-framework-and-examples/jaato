import { describe, expect, it } from "vitest";
import { countFiles, countHiddenFiles, effectiveHidden, entryId, hideToggleId, isDefaultHidden, isHidden } from "./WorkspacePanel";

describe("Files panel hide set", () => {
  it("uses the TUI's entry ids: a directory carries a trailing slash", () => {
    expect(entryId({ path: "src", children: { size: 2 } })).toBe("src/");
    expect(entryId({ path: "src/app.py", children: { size: 0 } })).toBe("src/app.py");
  });
  it("a hidden directory hides its subtree and nothing that merely shares a prefix", () => {
    expect(isHidden("src/app.py", ["src/"])).toBe(true);
    expect(isHidden("src/", ["src/"])).toBe(true);
    expect(isHidden("srcx/app.py", ["src/"])).toBe(false);
    expect(isHidden("src/app.py", ["src/app.pyc"])).toBe(false);
  });
  it("counts the files a hide set removes from view", () => {
    expect(countHiddenFiles({ "a/b.py": "created", "a/c.py": "modified", "d.py": "modified" }, ["a/"])).toBe(2);
  });
});

describe(".jaato/ is hidden by default (#1304 §5, a view filter, not .gitignore)", () => {
  it("is default-hidden itself and everything beneath it", () => {
    expect(isDefaultHidden(".jaato/")).toBe(true);
    expect(isDefaultHidden(".jaato/sessions/1.json")).toBe(true);
    expect(isDefaultHidden("src/app.py")).toBe(false);
  });
  it("effectiveHidden folds the default in without anything in the explicit hidden set", () => {
    expect(effectiveHidden(".jaato/logs/x.log", [])).toBe(true);
    expect(effectiveHidden("src/app.py", [])).toBe(false);
  });
  it("an explicit hide still hides an ordinary entry", () => {
    expect(effectiveHidden("src/", ["src/"])).toBe(true);
  });
  it("a !id exemption shows a default-hidden entry without disturbing the default for its siblings", () => {
    expect(effectiveHidden(".jaato/logs/x.log", ["!.jaato/logs/x.log"])).toBe(false);
    expect(effectiveHidden(".jaato/logs/y.log", ["!.jaato/logs/x.log"])).toBe(true);
  });
  it("hideToggleId: unhiding a default-hidden entry writes its !id exemption, not a bare id", () => {
    expect(hideToggleId(".jaato/logs/x.log", [])).toBe("!.jaato/logs/x.log");
  });
  it("hideToggleId: toggling again removes the exemption, back to default-hidden", () => {
    expect(hideToggleId(".jaato/logs/x.log", ["!.jaato/logs/x.log"])).toBe("!.jaato/logs/x.log");
  });
  it("hideToggleId: an ordinary entry still toggles its own id", () => {
    expect(hideToggleId("src/", [])).toBe("src/");
    expect(hideToggleId("src/", ["src/"])).toBe("src/");
  });
  it("counts .jaato/ entries in the hidden total even with no explicit hide", () => {
    expect(countHiddenFiles({ ".jaato/logs/a.log": "created", "d.py": "modified" }, [])).toBe(1);
  });
});

describe("Files panel collapse", () => {
  it("a folded directory reports every file beneath it, however deep", () => {
    const leaf = { children: new Map() };
    const sub = { children: new Map([["x", leaf], ["y", leaf]]) };
    const dir = { children: new Map<string, unknown>([["a", leaf], ["sub", sub]]) };
    expect(countFiles(dir as never)).toBe(3);
  });
});
