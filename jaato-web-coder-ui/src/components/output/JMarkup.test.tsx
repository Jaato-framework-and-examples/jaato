/**
 * Notebook cells render as cells, not as their tags (#1193).
 *
 * Reported from a deployed client: `<nb-row type="input" label="In [1]:">`
 * and `</nb-row>` showed as plain text around a correctly highlighted code
 * block, and the traceback below them as unstyled prose.  The code block
 * rendered because `parseJMarkup` knows `<j-code>`; the wrapper leaked
 * because nothing in the web client knew `<nb-row>` at all.
 *
 * `CELL` is not written by hand.  It is the daemon's own output for one cell
 * -- the notebook plugin's `_format_*_cell` emitters run through the real
 * formatter pipeline -- so this suite tests the wire the client actually
 * receives.  A hand-written fixture is how a mock ends up speaking the
 * client's vocabulary instead of the daemon's, which is how every earlier
 * rendering defect in this client stayed green.  (It even carries a literal
 * `<cell>` inside the traceback, which a real one does.)
 */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render } from "@testing-library/react";
import { hasServerMarkup, JMarkup } from "./JMarkup";

const CELL =
  '<nb-row type="input" label="In [1]:">\n<j-code language="ipython">\n<j-line n="1">print(1/0)</j-line>\n</j-code>\n\n</nb-row>\n' +
  '<nb-row type="stdout" label="Out [1]:">\nbefore the error\n</nb-row>\n' +
  '<nb-row type="error" label="Err [1]:">\nTraceback (most recent call last):\n  File "<cell>", line 1\nZeroDivisionError: division by zero\n</nb-row>\n';

// An early exit on the notebook's streaming path, after the formatter fix:
// no execution count, so no number in the label.
const EARLY_EXIT = '<nb-row type="error" label="Err:">\nNo code provided\n</nb-row>\n';

afterEach(cleanup);

describe("JMarkup renders notebook cells", () => {
  it("shows no wrapper tag anywhere -- the reported leak", () => {
    const { container } = render(<JMarkup text={CELL} />);
    expect(container.textContent).not.toMatch(/<\/?nb-row/);
  });

  it("draws one grid for the cell, one row per part, labels in their own column", () => {
    const { container } = render(<JMarkup text={CELL} />);
    expect(container.querySelectorAll(".nb-cells")).toHaveLength(1);
    const rows = [...container.querySelectorAll(".nb-row")];
    expect(rows.map((r) => r.getAttribute("data-nb-type"))).toEqual(["input", "stdout", "error"]);
    expect(rows.map((r) => r.querySelector(".nb-label")?.textContent)).toEqual(["In [1]:", "Out [1]:", "Err [1]:"]);
  });

  it("still highlights the code inside an input cell", () => {
    const { container } = render(<JMarkup text={CELL} />);
    const input = container.querySelector('.nb-row[data-nb-type="input"]');
    expect(input?.querySelector("pre.code-block")?.textContent).toContain("print(1/0)");
    expect(input?.textContent).toContain("ipython");
  });

  it("prints a traceback verbatim, indentation and angle brackets included", () => {
    const { container } = render(<JMarkup text={CELL} />);
    const error = container.querySelector('.nb-row[data-nb-type="error"]');
    expect(error?.classList.contains("nb-row-failed")).toBe(true);
    expect(error?.querySelector("pre.nb-out")?.textContent).toBe(
      'Traceback (most recent call last):\n  File "<cell>", line 1\nZeroDivisionError: division by zero',
    );
  });

  it("does not mark an ordinary output as a failure", () => {
    const { container } = render(<JMarkup text={CELL} />);
    expect(container.querySelector('.nb-row[data-nb-type="stdout"]')?.classList.contains("nb-row-failed")).toBe(false);
  });

  it("renders an early-exit error, which carries no execution count", () => {
    const { container } = render(<JMarkup text={EARLY_EXIT} />);
    expect(container.textContent).not.toMatch(/<\/?(nb-row|notebook-cell)/);
    expect(container.querySelector(".nb-label")?.textContent).toBe("Err:");
    expect(container.querySelector("pre.nb-out")?.textContent).toBe("No code provided");
  });

  it("keeps prose around a cell on the prose path", () => {
    const { container } = render(<JMarkup text={`I ran **one** cell:\n${CELL}That failed.`} />);
    expect(container.querySelector("strong")?.textContent).toBe("one");
    expect(container.textContent).toContain("That failed.");
    expect(container.querySelectorAll(".nb-cells")).toHaveLength(1);
  });

  it("the control: text with no notebook markup renders exactly as before", () => {
    const { container } = render(<JMarkup text={'Plain **bold**.\n<j-table>\n<j-tr><j-td>1</j-td></j-tr>\n</j-table>\n'} />);
    expect(container.querySelector(".nb-cells")).toBeNull();
    expect(container.querySelector("table.j-table td")?.textContent).toBe("1");
    expect(container.querySelector("strong")?.textContent).toBe("bold");
  });
});

describe("numbered code lines never become copyable text (jaato/#1304)", () => {
  // Two ``<j-line n="...">`` entries starting mid-file (line 5), so the
  // reported bug -- copying stdout yields the line numbers mixed in with
  // it -- has something to catch: if the number were still a text node,
  // "5" and "6" would show up in ``textContent`` right next to content
  // that itself contains none.
  const CODE = '<j-code language="python">\n<j-line n="5">print(0)</j-line>\n<j-line n="6">print(1)</j-line>\n</j-code>';

  it("puts the number nowhere DOM text (and so a copy) can reach it", () => {
    const { container } = render(<JMarkup text={CODE} />);
    expect(container.textContent).toContain("print(0)");
    expect(container.textContent).toContain("print(1)");
    // jsdom does not render ::before content into textContent at all --
    // which is the point: there is no text node for a copy to pick up.
    expect(container.textContent).not.toContain("5");
    expect(container.textContent).not.toContain("6");
  });

  it("still carries the exact number, as a CSS counter reset on the line itself", () => {
    const { container } = render(<JMarkup text={CODE} />);
    const lines = [...container.querySelectorAll(".code-block > div")];
    expect(lines).toHaveLength(2);
    expect(lines.every((l) => l.classList.contains("cl-n"))).toBe(true);
    // n=5 -> counter-reset to 4, so one increment (the CSS rule) reads 5.
    expect((lines[0] as HTMLElement).style.getPropertyValue("--cl-n")).toBe("4");
    expect((lines[1] as HTMLElement).style.getPropertyValue("--cl-n")).toBe("5");
  });

  it("an unnumbered code line gets no counter at all", () => {
    const { container } = render(<JMarkup text={'<j-code language="text">\n<j-line>no number here</j-line>\n</j-code>'} />);
    expect(container.querySelector("pre.code-block")?.textContent).toContain("no number here");
    expect(container.querySelector(".cl-n")).toBeNull();
  });
});

describe("hasServerMarkup -- what sends tool output here rather than to a raw <pre>", () => {
  it("claims a notebook cell that holds no <j-*> block at all", () => {
    // `print(42)`: the whole output is one stdout row.  The old gate asked
    // only about "<j-", so this went to the <pre> with its tags showing.
    const printed = '<nb-row type="stdout" label="Out [1]:">\n42\n</nb-row>\n';
    expect(printed.includes("<j-")).toBe(false);
    expect(hasServerMarkup(printed)).toBe(true);
  });

  it("still claims <j-*> and still declines plain output", () => {
    expect(hasServerMarkup('<j-code language="py">\n</j-code>')).toBe(true);
    expect(hasServerMarkup("total 12\ndrwxr-xr-x  2 me me 4096 .")).toBe(false);
  });
});
