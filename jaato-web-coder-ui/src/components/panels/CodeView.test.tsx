/**
 * The Files viewer's code view highlights by file name and renders the
 * result as elements -- the file's text must reach the page as text, so a
 * file holding markup shows the markup rather than becoming it.
 */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import CodeView, { MAX_HIGHLIGHT_CHARS } from "./CodeView";
import { languageForPath } from "@/protocol/codeLanguages";

afterEach(cleanup);

describe("languageForPath", () => {
  it("names a grammar by file name, then extension, and nothing for markdown or prose", () => {
    expect(languageForPath(".jaato/sessions/x.json")).toBe("json");
    expect(languageForPath("src/a.tsx")).toBe("typescript");
    expect(languageForPath("Dockerfile")).toBe("dockerfile");
    expect(languageForPath("pyproject.toml")).toBe("ini");
    expect(languageForPath("README.md")).toBeNull();
    expect(languageForPath("notes.txt")).toBeNull();
  });
});

describe("CodeView", () => {
  it("highlights JSON with hljs spans and numbers the lines", () => {
    const { container } = render(<CodeView text={'{\n  "a": 1\n}\n'} path="x.json" heightClass="max-h-64" />);
    expect(screen.getByTestId("code-view")).toHaveAttribute("data-language", "json");
    expect(container.querySelector("code .hljs-number")?.textContent).toBe("1");
    expect(screen.getByText("json · 3 lines")).toBeInTheDocument();
  });

  it("shows markup in a file as text, never as elements", () => {
    const { container } = render(<CodeView text={'<script>alert(1)</script>\n<img src=x onerror=alert(1)>'} path="page.html" heightClass="max-h-64" />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("code")?.textContent).toContain("<script>alert(1)</script>");
  });

  it("leaves a file too large to highlight plain, and says so", () => {
    const { container } = render(<CodeView text={"x".repeat(MAX_HIGHLIGHT_CHARS + 1)} path="big.py" heightClass="max-h-64" />);
    expect(container.querySelector("code span")).toBeNull();
    expect(screen.getByText("too large to highlight")).toBeInTheDocument();
  });

  it("wraps on request and drops the gutter while wrapped", () => {
    const { container } = render(<CodeView text={"a = 1\n"} path="a.py" heightClass="max-h-64" />);
    expect(container.querySelectorAll("pre")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "wrap" }));
    expect(container.querySelectorAll("pre")).toHaveLength(1);
  });
});
