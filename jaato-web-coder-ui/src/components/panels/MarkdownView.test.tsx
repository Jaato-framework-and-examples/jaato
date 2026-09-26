/**
 * The Files panel's markdown view renders a workspace document without
 * letting it do anything a document should not.
 *
 * The documents are often ones the agent wrote, so the cases that matter
 * are the refusals: raw HTML must not become elements, a ``javascript:``
 * link must not become a link, a remote image must not be fetched, and a
 * relative reference must go through the daemon (``fetchFile`` / ``onOpen``)
 * rather than the browser.  The positive cases are there to show the
 * refusals are not passing because nothing renders at all.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import MarkdownView from "./MarkdownView";

// mermaid needs a real layout engine (jsdom has no getBBox); the e2e suite
// draws a real diagram.  Here the module is replaced so the test sees what
// the view hands it and what it does with the answer.
const mermaidMock = vi.hoisted(() => ({
  initialize: vi.fn(),
  render: vi.fn(async (_id: string, src: string) => {
    if (src.includes("bogus")) throw new Error("Parse error on line 1");
    return { svg: `<svg xmlns="http://www.w3.org/2000/svg"><text>${src}</text></svg>` };
  }),
}));
vi.mock("mermaid", () => ({ default: mermaidMock }));

afterEach(cleanup);

const noFetch = () => Promise.resolve(null);

function view(source: string, extra: Partial<Parameters<typeof MarkdownView>[0]> = {}) {
  return render(<MarkdownView source={source} path="docs/index.md" fetchFile={noFetch} onOpen={() => undefined} {...extra} />);
}

describe("MarkdownView renders GitHub-flavoured markdown", () => {
  it("renders headings, tables, task lists and highlighted code", () => {
    const { container } = view([
      "# Title",
      "",
      "| a | b |",
      "|---|---|",
      "| 1 | 2 |",
      "",
      "- [x] done",
      "- [ ] todo",
      "",
      "```python",
      "def f(): return 1",
      "```",
    ].join("\n"));
    expect(screen.getByRole("heading", { level: 1, name: "Title" })).toHaveAttribute("id", "md-title");
    expect(screen.getByRole("table")).toHaveClass("j-table");
    expect(screen.getAllByRole("checkbox")).toHaveLength(2);
    expect(container.querySelector("pre code.language-python .hljs-keyword")?.textContent).toBe("def");
  });

  it("highlights a fence by alias and leaves an unregistered or missing language plain", () => {
    const { container } = view("```sh\necho hi\n```\n\n```cobol\nMOVE A TO B\n```\n\n```\nplain\n```");
    const codes = container.querySelectorAll("pre code");
    expect(codes).toHaveLength(3);
    expect(codes[0]!.querySelector("[class^='hljs-']")).not.toBeNull();
    expect(codes[1]!.querySelector("[class^='hljs-']")).toBeNull();
    expect(codes[1]!.textContent).toBe("MOVE A TO B\n");
    expect(codes[2]!.querySelector("span")).toBeNull();
  });
});

describe("MarkdownView refuses what a document should not do", () => {
  it("drops raw HTML instead of rendering it", () => {
    const { container } = view("before\n\n<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\nafter");
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText("after")).toBeInTheDocument();
  });

  it("renders a javascript: link as text, not a link", () => {
    view("[click me](javascript:alert(1))");
    expect(screen.queryByRole("link", { name: "click me" })).toBeNull();
    expect(screen.getByText("click me")).toBeInTheDocument();
  });

  it("opens web links in a new tab without an opener", () => {
    view("[site](https://example.com)");
    const a = screen.getByRole("link", { name: "site" });
    expect(a).toHaveAttribute("href", "https://example.com");
    expect(a).toHaveAttribute("target", "_blank");
    expect(a.getAttribute("rel")).toContain("noopener");
  });

  it("does not load a remote image until asked, then loads every one without a referrer", () => {
    const fetchFile = vi.fn(noFetch);
    const { container } = view("![logo](https://tracker.example/p.png)\n\n![badge](https://ci.example/b.svg)", { fetchFile });
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText(/remote image: logo · tracker\.example/)).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "load remote images" })[0]!);
    const imgs = screen.getAllByRole("img");
    expect(imgs.map((i) => i.getAttribute("src"))).toEqual(["https://tracker.example/p.png", "https://ci.example/b.svg"]);
    for (const i of imgs) expect(i).toHaveAttribute("referrerpolicy", "no-referrer");
    expect(fetchFile).not.toHaveBeenCalled();
  });
});

describe("MarkdownView resolves workspace references through the daemon", () => {
  it("a relative link opens the resolved workspace file in the viewer", () => {
    const onOpen = vi.fn();
    view("see [setup](../SETUP.md#step-2)", { onOpen });
    fireEvent.click(screen.getByRole("link", { name: "setup" }));
    expect(onOpen).toHaveBeenCalledWith("SETUP.md");
  });

  it("a relative image is fetched by its workspace path and shown from a blob URL", async () => {
    const createObjectURL = vi.fn(() => "blob:fake");
    const revokeObjectURL = vi.fn();
    Object.assign(URL, { createObjectURL, revokeObjectURL });
    const fetchFile = vi.fn(() => Promise.resolve(new Uint8Array([137, 80, 78, 71])));
    const { unmount } = view("![diagram](img/arch.png)", { fetchFile });
    await waitFor(() => expect(screen.getByRole("img", { name: "diagram" })).toHaveAttribute("src", "blob:fake"));
    expect(fetchFile).toHaveBeenCalledWith("docs/img/arch.png");
    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:fake");
  });

  it("a relative image the daemon refuses is named, not shown broken", async () => {
    view("![secret](../../etc/x.png) ![gone](missing.png)");
    expect(await screen.findByText("[image: gone]")).toBeInTheDocument();
    expect(screen.getByText("[image: secret]")).toBeInTheDocument();
  });
});

describe("MarkdownView draws mermaid fences", () => {
  it("renders a mermaid fence as an image, with mermaid's strict security level", async () => {
    Object.assign(URL, { createObjectURL: vi.fn(() => "blob:diagram"), revokeObjectURL: vi.fn() });
    const { container } = view("```mermaid\ngraph TD; A-->B\n```");
    const img = await screen.findByRole("img", { name: "Mermaid diagram" });
    expect(img).toHaveAttribute("src", "blob:diagram");
    // The SVG is never inserted as markup: no <svg> in the document.
    expect(container.querySelector("svg")).toBeNull();
    expect(mermaidMock.initialize).toHaveBeenCalledWith(expect.objectContaining({ securityLevel: "strict", htmlLabels: false }));
    expect(mermaidMock.render).toHaveBeenCalledWith(expect.any(String), "graph TD; A-->B\n");
  });

  it("shows the error and the source when the diagram does not parse", async () => {
    view("```mermaid\nbogus diagram\n```");
    expect(await screen.findByRole("alert")).toHaveTextContent("Parse error on line 1");
    expect(screen.getByText("bogus diagram")).toBeInTheDocument();
  });
});

describe("naturalWidthOf", () => {
  it("reads the width mermaid drew at, and nothing from an SVG that states none", async () => {
    const { naturalWidthOf } = await import("./MermaidDiagram");
    expect(naturalWidthOf('<svg width="100%" style="max-width: 130.5px;" viewBox="0 0 130 300">')).toBe(131);
    expect(naturalWidthOf("<svg viewBox='0 0 1 1'>")).toBeNull();
  });
});
