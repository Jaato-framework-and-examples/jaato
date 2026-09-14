/**
 * Parser for the server's client-neutral ``<j-*>`` markup.
 *
 * The daemon's formatter pipeline never renders to a terminal or to
 * HTML; it emits semantic tags that every attached client renders in
 * its own idiom (``jaato-tui/j_markup_renderer.py`` turns them into
 * ANSI, this module turns them into a small AST the React layer turns
 * into DOM).  Keeping the wire neutral is what lets a TUI and a browser
 * co-attach to one session without fighting over the format.
 *
 * Grammar (emitted by ``code_block_formatter`` and ``table_formatter``):
 *
 *   <j-code language="py">\n<j-line n="1"><j-tok t="k">def</j-tok> f</j-line>\n</j-code>
 *   <j-table>\n<j-thead><j-tr><j-th>a</j-th></j-tr></j-thead>\n<j-tr><j-td>1</j-td></j-tr>\n</j-table>
 *
 * Text inside tags is HTML-escaped (``&amp; &lt; &gt; &quot;``) so a
 * ``<`` in source code cannot collide with the tag namespace; this
 * module unescapes exactly those entities.  Everything outside a
 * ``<j-code>``/``<j-table>`` block is passed through as ``text`` and
 * is the pipeline's markdown, rendered by ``markdown.ts``.
 *
 * The parser is tolerant: an unterminated block (mid-stream) is left
 * as text until its closing tag arrives, so a streaming renderer can
 * re-parse the growing buffer on each chunk without flicker.
 */

export interface Token {
  /** Pygments short token name (``k``, ``nf``, ``s2``…); ``""`` for plain text. */
  t: string;
  text: string;
}

export interface CodeLine {
  n: number | null;
  tokens: Token[];
}

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "code"; language: string; lines: CodeLine[] }
  | { kind: "table"; head: string[]; rows: string[][] };

// Every ``/g`` regex is instantiated per call (``re()``): the parsers nest
// (block → line → token), and a shared global regex carries ``lastIndex``
// across calls, which is how a nested exec loop turns into an infinite one.
const BLOCK_SRC = '<j-code(?:\\s+language="([^"]*)")?>\\n?([\\s\\S]*?)<\\/j-code>\\n?|<j-table>\\n?([\\s\\S]*?)<\\/j-table>\\n?';
const LINE_SRC = '<j-line(?:\\s+n="(\\d+)")?>([\\s\\S]*?)<\\/j-line>';
const TOK_SRC = '<j-tok(?:\\s+t="([^"]*)")?>([\\s\\S]*?)<\\/j-tok>';
const TR_SRC = "<j-tr>([\\s\\S]*?)<\\/j-tr>";
const CELL_SRC = "<j-t[hd]>([\\s\\S]*?)<\\/j-t[hd]>";
const THEAD_RE = /<j-thead>\n?([\s\S]*?)<\/j-thead>/;
const re = (src: string) => new RegExp(src, "g");

export function unescapeMarkup(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}

/** Does the buffer contain an opened block whose close has not arrived yet? */
export function hasOpenBlock(text: string): boolean {
  const lastCodeOpen = text.lastIndexOf("<j-code");
  const lastCodeClose = text.lastIndexOf("</j-code>");
  if (lastCodeOpen > lastCodeClose) return true;
  const lastTableOpen = text.lastIndexOf("<j-table>");
  const lastTableClose = text.lastIndexOf("</j-table>");
  return lastTableOpen > lastTableClose;
}

function parseLine(body: string): Token[] {
  const tokens: Token[] = [];
  const TOK_RE = re(TOK_SRC);
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = TOK_RE.exec(body)) !== null) {
    if (m.index > last) tokens.push({ t: "", text: unescapeMarkup(body.slice(last, m.index)) });
    tokens.push({ t: m[1] ?? "", text: unescapeMarkup(m[2] ?? "") });
    last = m.index + m[0].length;
  }
  if (last < body.length) tokens.push({ t: "", text: unescapeMarkup(body.slice(last)) });
  return tokens;
}

export function parseCodeBody(body: string): CodeLine[] {
  const lines: CodeLine[] = [];
  const LINE_RE = re(LINE_SRC);
  let m: RegExpExecArray | null;
  let any = false;
  while ((m = LINE_RE.exec(body)) !== null) {
    any = true;
    lines.push({ n: m[1] ? Number(m[1]) : null, tokens: parseLine(m[2] ?? "") });
  }
  if (!any) {
    // A block without <j-line> wrappers (defensive): one line per newline.
    for (const raw of body.replace(/\n$/, "").split("\n")) {
      lines.push({ n: null, tokens: [{ t: "", text: unescapeMarkup(raw) }] });
    }
  }
  return lines;
}

function parseCells(row: string): string[] {
  const cells: string[] = [];
  const CELL_RE = re(CELL_SRC);
  let m: RegExpExecArray | null;
  while ((m = CELL_RE.exec(row)) !== null) cells.push(unescapeMarkup((m[1] ?? "").trim()));
  return cells;
}

export function parseTableBody(body: string): { head: string[]; rows: string[][] } {
  let head: string[] = [];
  let rest = body;
  const th = THEAD_RE.exec(body);
  if (th) {
    const headRows = th[1] ?? "";
    const first = re(TR_SRC).exec(headRows);
    head = parseCells(first ? (first[1] ?? "") : headRows);
    rest = body.slice(th.index + th[0].length);
  }
  const rows: string[][] = [];
  const TR_RE = re(TR_SRC);
  let m: RegExpExecArray | null;
  while ((m = TR_RE.exec(rest)) !== null) rows.push(parseCells(m[1] ?? ""));
  return { head, rows };
}

/** Split a buffer into text / code / table segments. */
export function parseJMarkup(text: string): Segment[] {
  const out: Segment[] = [];
  const BLOCK_RE = re(BLOCK_SRC);
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = BLOCK_RE.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    if (m[3] !== undefined) {
      out.push({ kind: "table", ...parseTableBody(m[3]) });
    } else {
      out.push({ kind: "code", language: unescapeMarkup(m[1] ?? ""), lines: parseCodeBody(m[2] ?? "") });
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/** Plain-text projection (for copy-to-clipboard and search). */
export function jmarkupToPlainText(text: string): string {
  return parseJMarkup(text)
    .map((seg) => {
      if (seg.kind === "text") return seg.text;
      if (seg.kind === "code") return seg.lines.map((l) => l.tokens.map((t) => t.text).join("")).join("\n") + "\n";
      const rows = [seg.head, ...seg.rows].filter((r) => r.length);
      return rows.map((r) => r.join(" | ")).join("\n") + "\n";
    })
    .join("");
}
