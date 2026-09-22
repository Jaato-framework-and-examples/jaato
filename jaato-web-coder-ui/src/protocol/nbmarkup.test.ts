import { describe, expect, it } from "vitest";
import { containsNbMarkup, isFailureRow, parseNbMarkup } from "./nbmarkup";
import { parseJMarkup } from "./jmarkup";

// What the daemon actually sends for one executed cell: the notebook plugin's
// own <nb-row> emitters, with the input fence already turned into <j-code> by
// the formatter pipeline (shared/plugins/notebook/plugin.py _format_*_cell).
const CELL = [
  '<nb-row type="input" label="In [1]:">',
  '<j-code language="ipython">',
  '<j-line><j-tok t="k">print</j-tok>(42)</j-line>',
  "</j-code>",
  "</nb-row>",
  '<nb-row type="stdout" label="Out [1]:">',
  "42",
  "</nb-row>",
  '<nb-row type="error" label="Err [1]:">',
  "Traceback (most recent call last):",
  "  File \"<cell>\", line 1",
  "ZeroDivisionError: division by zero",
  "</nb-row>",
].join("\n");

describe("parseNbMarkup", () => {
  it("splits a cell into its rows, keeping each row's type and label", () => {
    const rows = parseNbMarkup(CELL).filter((s) => s.kind === "row");
    expect(rows.map((s) => s.kind === "row" && [s.row.type, s.row.label])).toEqual([
      ["input", "In [1]:"],
      ["stdout", "Out [1]:"],
      ["error", "Err [1]:"],
    ]);
  });

  it("leaves no wrapper tag in any text segment -- the reported leak", () => {
    for (const seg of parseNbMarkup(`Ran it:\n${CELL}\nDone.`)) {
      if (seg.kind === "text") expect(seg.text).not.toMatch(/<\/?nb-row/);
    }
  });

  it("hands a row's body back raw, so <j-code> inside a cell still parses", () => {
    const input = parseNbMarkup(CELL)[0];
    if (input?.kind !== "row") throw new Error("expected a row");
    expect(input.row.content.startsWith('<j-code language="ipython">')).toBe(true);
    expect(parseJMarkup(input.row.content).map((s) => s.kind)).toEqual(["code"]);
  });

  it("strips exactly the markup's own newlines and keeps indentation", () => {
    const err = parseNbMarkup(CELL)[2];
    if (err?.kind !== "row") throw new Error("expected a row");
    expect(err.row.content).toBe(
      'Traceback (most recent call last):\n  File "<cell>", line 1\nZeroDivisionError: division by zero',
    );
  });

  it("accepts the formatter's shapes: an empty label, and an error with no count", () => {
    const [stdout, early] = parseNbMarkup(
      '<nb-row type="stdout" label="">\nhi\n</nb-row>\n<nb-row type="error" label="Err:">\nNo code provided\n</nb-row>\n',
    );
    expect(stdout).toEqual({ kind: "row", row: { type: "stdout", label: "", content: "hi" } });
    expect(early).toEqual({ kind: "row", row: { type: "error", label: "Err:", content: "No code provided" } });
  });

  it("keeps the text around rows", () => {
    const segs = parseNbMarkup(`Before.\n${CELL}\nAfter.`);
    expect(segs[0]).toEqual({ kind: "text", text: "Before.\n" });
    expect(segs.at(-1)).toEqual({ kind: "text", text: "After." });
  });

  it("leaves an unterminated row as text, as jmarkup leaves an unterminated block", () => {
    // A model quoting the tag in prose must not have the rest of its
    // message swallowed into a cell that does not exist.
    const quoted = 'The server writes <nb-row type="input" label="In [1]:"> before each cell.';
    expect(parseNbMarkup(quoted)).toEqual([{ kind: "text", text: quoted }]);
    expect(containsNbMarkup(quoted)).toBe(false);
  });

  it("is a single text segment when there is no notebook markup", () => {
    expect(parseNbMarkup("plain text")).toEqual([{ kind: "text", text: "plain text" }]);
    expect(containsNbMarkup("plain <j-code>x</j-code>")).toBe(false);
    expect(containsNbMarkup(CELL)).toBe(true);
  });

  it("treats error and stderr rows as failures, and nothing else", () => {
    const kinds = ["input", "stdout", "stderr", "result", "display", "error"];
    const failing = kinds.filter((type) => isFailureRow({ type, label: "", content: "" }));
    expect(failing).toEqual(["stderr", "error"]);
  });
});

describe("the two parsers stay bounded to their own tags", () => {
  // The TUI pins the same boundary (test_nb_row_is_not_j_markup): each
  // parser knows one family, and the renderer composes them.
  it("parseJMarkup does not claim <nb-row>", () => {
    expect(parseJMarkup('<nb-row type="stdout" label="">\nx\n</nb-row>')).toEqual([
      { kind: "text", text: '<nb-row type="stdout" label="">\nx\n</nb-row>' },
    ]);
  });
});
