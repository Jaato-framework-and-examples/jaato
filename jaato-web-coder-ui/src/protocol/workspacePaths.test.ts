import { describe, expect, it } from "vitest";
import { classifyReference, imageMimeFor, isMarkdownPath, resolveWorkspacePath } from "./workspacePaths";

describe("isMarkdownPath", () => {
  it("recognises the markdown extensions, case-insensitively", () => {
    for (const p of ["README.md", "docs/x.MD", "a.markdown", "b.mdx"]) expect(isMarkdownPath(p)).toBe(true);
    for (const p of ["notes.txt", "md", "x.md.bak", "src/app.py"]) expect(isMarkdownPath(p)).toBe(false);
  });
});

describe("resolveWorkspacePath", () => {
  it("resolves against the document's directory", () => {
    expect(resolveWorkspacePath("setup.md", "docs/index.md")).toBe("docs/setup.md");
    expect(resolveWorkspacePath("./img/a.png", "docs/index.md")).toBe("docs/img/a.png");
    expect(resolveWorkspacePath("../README.md", "docs/index.md")).toBe("README.md");
  });
  it("reads a leading slash as the workspace root, as GitHub does", () => {
    expect(resolveWorkspacePath("/CONTRIBUTING.md", "docs/deep/x.md")).toBe("CONTRIBUTING.md");
  });
  it("refuses a reference that climbs above the root", () => {
    expect(resolveWorkspacePath("../../etc/passwd", "docs/x.md")).toBeNull();
    expect(resolveWorkspacePath("../x", "README.md")).toBeNull();
    expect(resolveWorkspacePath("/../x", "README.md")).toBeNull();
  });
});

describe("classifyReference", () => {
  it("opens web links externally and blocks every other scheme", () => {
    expect(classifyReference("https://example.com/a", "README.md")).toEqual({ kind: "external", href: "https://example.com/a" });
    expect(classifyReference("mailto:a@b.c", "README.md").kind).toBe("external");
    for (const h of ["javascript:alert(1)", "JavaScript:alert(1)", "data:text/html,x", "file:///etc/passwd", "//evil.example/x", ""]) {
      expect(classifyReference(h, "README.md")).toEqual({ kind: "blocked" });
    }
  });
  it("keeps an in-document anchor as an anchor", () => {
    expect(classifyReference("#install", "README.md")).toEqual({ kind: "anchor", fragment: "install" });
  });
  it("resolves a workspace path, dropping the query and keeping the fragment", () => {
    expect(classifyReference("docs/setup%20guide.md?x=1#step-2", "README.md")).toEqual({ kind: "workspace", path: "docs/setup guide.md", fragment: "step-2" });
  });
  it("blocks a path that escapes the workspace, and one that will not decode", () => {
    expect(classifyReference("../../secret", "docs/x.md")).toEqual({ kind: "blocked" });
    expect(classifyReference("bad%E0%A4%A.md", "x.md")).toEqual({ kind: "blocked" });
  });
});

describe("imageMimeFor", () => {
  it("types the image formats a browser renders and nothing else", () => {
    expect(imageMimeFor("a/b.PNG")).toBe("image/png");
    expect(imageMimeFor("d.svg")).toBe("image/svg+xml");
    expect(imageMimeFor("notes.md")).toBeNull();
    expect(imageMimeFor("noext")).toBeNull();
  });
});
