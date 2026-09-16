import { describe, expect, it } from "vitest";
import { parseInline, parseMarkdown } from "./markdown";

describe("markdown", () => {
  it("parses inline code, emphasis and links", () => {
    expect(parseInline("a `b` **c** _d_ [e](https://x.y)")).toEqual([
      { kind: "text", text: "a " },
      { kind: "code", text: "b" },
      { kind: "text", text: " " },
      { kind: "strong", children: [{ kind: "text", text: "c" }] },
      { kind: "text", text: " " },
      { kind: "em", children: [{ kind: "text", text: "d" }] },
      { kind: "text", text: " " },
      { kind: "link", href: "https://x.y", children: [{ kind: "text", text: "e" }] },
    ]);
  });
  it("does not treat snake_case as emphasis", () => {
    expect(parseInline("call foo_bar_baz now")).toEqual([{ kind: "text", text: "call foo_bar_baz now" }]);
  });
  it("parses blocks", () => {
    const b = parseMarkdown("# T\n\npara one\nstill\n\n- a\n- b\n\n1. x\n\n> q\n\n---\n");
    expect(b.map((x) => x.kind)).toEqual(["heading", "paragraph", "list", "list", "quote", "rule"]);
    expect(b[2]).toMatchObject({ ordered: false });
    expect(b[3]).toMatchObject({ ordered: true });
  });
});
