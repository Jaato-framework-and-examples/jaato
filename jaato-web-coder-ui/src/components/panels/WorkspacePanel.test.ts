import { describe, expect, it } from "vitest";
import { countFiles, countHiddenFiles, entryId, isHidden } from "./WorkspacePanel";

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

describe("Files panel collapse", () => {
  it("a folded directory reports every file beneath it, however deep", () => {
    const leaf = { children: new Map() };
    const sub = { children: new Map([["x", leaf], ["y", leaf]]) };
    const dir = { children: new Map<string, unknown>([["a", leaf], ["sub", sub]]) };
    expect(countFiles(dir as never)).toBe(3);
  });
});
