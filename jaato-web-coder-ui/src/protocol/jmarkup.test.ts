import { describe, expect, it } from "vitest";
import { hasOpenBlock, jmarkupToPlainText, parseJMarkup } from "./jmarkup";

const CODE =
  'Here:\n<j-code language="python">\n<j-line n="1"><j-tok t="k">def</j-tok> <j-tok t="nf">f</j-tok>():</j-line>\n<j-line n="2">    x = <j-tok t="s2">"&lt;a&amp;b&gt;"</j-tok></j-line>\n</j-code>\ndone';

describe("parseJMarkup", () => {
  it("splits text and code, unescaping entities inside tokens", () => {
    const segs = parseJMarkup(CODE);
    expect(segs.map((s) => s.kind)).toEqual(["text", "code", "text"]);
    const code = segs[1]!;
    if (code.kind !== "code") throw new Error("expected code");
    expect(code.language).toBe("python");
    expect(code.lines).toHaveLength(2);
    expect(code.lines[0]!.n).toBe(1);
    expect(code.lines[0]!.tokens).toEqual([
      { t: "k", text: "def" },
      { t: "", text: " " },
      { t: "nf", text: "f" },
      { t: "", text: "():" },
    ]);
    expect(code.lines[1]!.tokens.at(-1)).toEqual({ t: "s2", text: '"<a&b>"' });
  });
  it("parses tables with a header row", () => {
    const t = "<j-table>\n<j-thead><j-tr><j-th>Name</j-th><j-th>Qty</j-th></j-tr></j-thead>\n<j-tr><j-td>a</j-td><j-td>1</j-td></j-tr>\n<j-tr><j-td>b &amp; c</j-td><j-td>2</j-td></j-tr>\n</j-table>\n";
    const segs = parseJMarkup(t);
    expect(segs).toEqual([{ kind: "table", head: ["Name", "Qty"], rows: [["a", "1"], ["b & c", "2"]] }]);
  });
  it("leaves an unterminated block as text while streaming", () => {
    const partial = 'intro <j-code language="ts">\n<j-line n="1">const';
    expect(parseJMarkup(partial)).toEqual([{ kind: "text", text: partial }]);
    expect(hasOpenBlock(partial)).toBe(true);
    expect(hasOpenBlock(CODE)).toBe(false);
  });
  it("handles a bare <j-code> without language", () => {
    const segs = parseJMarkup("<j-code>\n<j-line>x</j-line>\n</j-code>");
    expect(segs[0]).toMatchObject({ kind: "code", language: "" });
  });
  it("projects to plain text", () => {
    expect(jmarkupToPlainText(CODE)).toBe('Here:\ndef f():\n    x = "<a&b>"\ndone');
  });
});
