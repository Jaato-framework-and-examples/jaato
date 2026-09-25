import { describe, expect, it } from "vitest";
import { diffPreviewForCall, execOutputPreview, execTitle } from "./toolPreview";

describe("diffPreviewForCall", () => {
  it("builds a before/after diff from updateFile's targeted old/new args", () => {
    const p = diffPreviewForCall("updateFile", { path: "a.py", old: "x = 1", new: "x = 2" });
    expect(p.path).toBe("a.py");
    expect(p.diffLines).toEqual(["-x = 1", "+x = 2"]);
    expect(p.text).toBeNull();
  });

  it("falls back to the new_content preview when updateFile rewrote the whole file", () => {
    const p = diffPreviewForCall("updateFile", { path: "a.py", new_content: "line1\nline2" });
    expect(p.diffLines).toBeNull();
    expect(p.text).toBe("line1\nline2");
  });

  it("previews a new file's content with no before to diff against", () => {
    const p = diffPreviewForCall("writeNewFile", { path: "new.py", content: "print(1)" });
    expect(p.diffLines).toBeNull();
    expect(p.text).toBe("print(1)");
  });

  it("caps a long write preview", () => {
    const p = diffPreviewForCall("writeNewFile", { path: "big.py", content: Array.from({ length: 20 }, (_, i) => `line ${i}`).join("\n") });
    expect(p.truncated).toBe(true);
    expect(p.text?.split("\n").length).toBeLessThanOrEqual(6);
  });

  it("summarises a structural change with no content to preview", () => {
    expect(diffPreviewForCall("removeFile", { path: "gone.py" }).text).toBe("removed gone.py");
    expect(diffPreviewForCall("moveFile", { path: "a.py", new_path: "b.py" }).text).toBe("a.py → b.py");
  });

  it("previews nothing for a tool it has no shape for", () => {
    const p = diffPreviewForCall("someOtherWrite", { path: "x" });
    expect(p.diffLines).toBeNull();
    expect(p.text).toBeNull();
  });
});

describe("execOutputPreview / execTitle", () => {
  it("previews the tail of long output", () => {
    const output = Array.from({ length: 20 }, (_, i) => `line ${i}`).join("\n");
    const { text, truncated } = execOutputPreview(output);
    expect(truncated).toBe(true);
    expect(text).toContain("line 19");
    expect(text).not.toContain("line 0\n");
  });

  it("does not report a short output as truncated", () => {
    expect(execOutputPreview("a\nb").truncated).toBe(false);
  });

  it("titles a shell call by its command, and falls back to the tool name", () => {
    expect(execTitle("cli_based_tool", { command: "git status" })).toBe("git status");
    expect(execTitle("shell_input", { text: "y\n" })).toBe("y\n");
    expect(execTitle("cli_based_tool", {})).toBe("cli_based_tool");
  });
});
