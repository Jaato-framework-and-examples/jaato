import { describe, expect, it } from "vitest";
import { countHiddenFiles, entryId, isHidden } from "./WorkspacePanel";

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
