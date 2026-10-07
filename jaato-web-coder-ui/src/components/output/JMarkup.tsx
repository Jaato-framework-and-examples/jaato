/**
 * Renders the server's markup as DOM: ``<nb-row>`` notebook cells,
 * ``<j-code>`` / ``<j-table>`` blocks, and the markdown between them.
 * Pure function of its ``text`` prop; memoised per block so a streaming
 * append re-renders only the block that grew.  Never uses ``innerHTML``.
 *
 * **Two parsers, composed here and only here.**  ``nbmarkup.ts`` knows
 * ``<nb-row>`` and nothing else; ``jmarkup.ts`` knows ``<j-*>`` and
 * nothing else (the boundary the TUI keeps between
 * ``j_markup_renderer.py`` and ``_render_notebook_rows``).  The buffer is
 * split on notebook rows FIRST, so a row's wrapper tags never reach the
 * ``<j-*>`` parser -- which, not knowing them, used to pass them through
 * as text: the leak in #1193.  A row's body then goes through
 * ``parseJMarkup`` in its turn, because an input cell carries a
 * ``<j-code>`` block and a printed markdown table arrives as ``<j-table>``.
 *
 * **A cell body is program output, not prose.**  Its plain text renders
 * verbatim (``.nb-out``: monospace, whitespace kept) rather than through
 * the markdown parser, which would read a traceback's ``*`` / ``_`` as
 * emphasis and reflow its indentation -- the TUI prints it verbatim too.
 */
import { Plate } from "@/components/layout/Plate";
import { memo, useMemo, type CSSProperties } from "react";
import { parseJMarkup, type CodeLine, type Segment } from "@/protocol/jmarkup";
import { containsNbMarkup, isFailureRow, parseNbMarkup, type NotebookRow, type NotebookSegment } from "@/protocol/nbmarkup";
import { parseMarkdown, type Block, type Inline } from "@/protocol/markdown";
import { tokenClass } from "@/protocol/pygments";

function Inlines({ nodes }: { nodes: Inline[] }) {
  return (
    <>
      {nodes.map((n, i) => {
        switch (n.kind) {
          case "text": return <span key={i}>{n.text}</span>;
          case "code": return <code key={i} className="inline">{n.text}</code>;
          case "strong": return <strong key={i}><Inlines nodes={n.children} /></strong>;
          case "em": return <em key={i}><Inlines nodes={n.children} /></em>;
          case "link": return <a key={i} href={n.href} target="_blank" rel="noreferrer noopener"><Inlines nodes={n.children} /></a>;
        }
      })}
    </>
  );
}

function MarkdownBlocks({ blocks }: { blocks: Block[] }) {
  return (
    <>
      {blocks.map((b, i) => {
        switch (b.kind) {
          case "paragraph": return <p key={i} className="whitespace-pre-wrap break-words"><Inlines nodes={b.children} /></p>;
          case "heading": {
            const Tag = (`h${Math.min(4, b.level)}`) as "h1" | "h2" | "h3" | "h4";
            return <Tag key={i}><Inlines nodes={b.children} /></Tag>;
          }
          case "list": {
            const Tag = b.ordered ? "ol" : "ul";
            return <Tag key={i}>{b.items.map((it, j) => <li key={j} className="whitespace-pre-wrap"><Inlines nodes={it} /></li>)}</Tag>;
          }
          case "quote": return <blockquote key={i} className="whitespace-pre-wrap"><Inlines nodes={b.children} /></blockquote>;
          case "rule": return <hr key={i} />;
          case "pre": return <pre key={i} className="code-block plate plate-ground p-2.5 my-1.5 overflow-x-auto whitespace-pre">{b.text}</pre>;
        }
      })}
    </>
  );
}

/**
 * A code line's number is CSS, never a text node -- jaato/#1304's
 * acceptance criterion is that copying stdout yields exactly the
 * stdout, and a literal ``<span>{n}</span>`` ahead of the tokens is one
 * more text node a selection spanning the block picks up if a browser's
 * ``user-select: none`` handling of a PROGRAMMATIC copy ever disagrees
 * with a mouse-drag one (a real, documented cross-browser
 * inconsistency, not a hypothetical one).  A ``::before`` pseudo-
 * element's ``content`` is never part of the DOM text a selection can
 * contain, in any engine, so there is nothing to leak.  ``l.n`` need not
 * start at 1 (an excerpt from mid-file), so each numbered line resets
 * the CSS counter to ``n - 1`` on ITSELF via ``--cl-n`` and increments
 * once (``.cl-n``'s rule in theme.css) -- a counter set and read on the
 * same element needs no running total shared across lines.
 */
function CodeLines({ lines }: { lines: CodeLine[] }) {
  return (
    <pre className="code-block overflow-x-auto whitespace-pre p-2">
      {lines.map((l, i) => (
        <div
          key={i}
          className={l.n !== null ? "cl-n" : undefined}
          style={l.n !== null ? ({ "--cl-n": (l.n as number) - 1 } as CSSProperties) : undefined}
        >
          {l.tokens.map((t, j) => (t.t ? <span key={j} className={tokenClass(t.t)}>{t.text}</span> : <span key={j}>{t.text}</span>))}
          {l.tokens.length === 0 && " "}
        </div>
      ))}
    </pre>
  );
}

export function CodeSegment({ language, lines }: { language: string; lines: CodeLine[] }) {
  const text = lines.map((l) => l.tokens.map((t) => t.text).join("")).join("\n");
  return (
    <Plate ground corners="two" className="my-2">
      <div className="flex items-center justify-between px-2.5 py-1 border-b hairline">
        <span className="kicker kicker-muted">{language || "text"}</span>
        <button
          type="button"
          className="chrome-sm font-heading uppercase tracking-[0.1em] text-text-muted hover:text-steel"
          onClick={() => navigator.clipboard?.writeText(text).catch(() => undefined)}
          title="Copy code"
        >
          copy
        </button>
      </div>
      <CodeLines lines={lines} />
    </Plate>
  );
}

function TableSegment({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="j-table">
        {head.length > 0 && (
          <thead><tr>{head.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
        )}
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function SegmentView({ seg }: { seg: Segment }) {
  if (seg.kind === "code") return <CodeSegment language={seg.language} lines={seg.lines} />;
  if (seg.kind === "table") return <TableSegment head={seg.head} rows={seg.rows} />;
  return <MarkdownBlocks blocks={parseMarkdown(seg.text)} />;
}

/** ``<j-*>`` + markdown, the path every non-notebook text has always taken. */
function JSegments({ text }: { text: string }) {
  return <>{parseJMarkup(text).map((seg, i) => <SegmentView key={i} seg={seg} />)}</>;
}

/** One row's body: code and tables as everywhere else, plain text verbatim. */
function CellBody({ row }: { row: NotebookRow }) {
  return (
    <>
      {parseJMarkup(row.content).map((seg, i) => {
        if (seg.kind !== "text") return <SegmentView key={i} seg={seg} />;
        const out = seg.text.replace(/^\n+|\n+$/g, "");
        return out.trim() ? <pre key={i} className="nb-out">{out}</pre> : null;
      })}
    </>
  );
}

/**
 * The rows of one notebook cell, laid out as the TUI draws them: a label
 * column (``In [1]:`` / ``Out [1]:`` / ``Err [1]:``) beside the body, one
 * grid for the whole cell so the labels line up.  ``error`` and ``stderr``
 * rows take the error tone; ``data-nb-type`` carries the row's type through
 * for anything that wants to style the rest.
 */
function NotebookCells({ rows }: { rows: NotebookRow[] }) {
  return (
    <div className="nb-cells">
      {rows.map((row, i) => (
        <div key={i} className={`nb-row${isFailureRow(row) ? " nb-row-failed" : ""}`} data-nb-type={row.type}>
          <span className="nb-label">{row.label}</span>
          <div className="nb-body min-w-0"><CellBody row={row} /></div>
        </div>
      ))}
    </div>
  );
}

type Piece = { kind: "text"; text: string } | { kind: "cells"; rows: NotebookRow[] };

/**
 * Consecutive rows are one cell: gather them into one grid, dropping the
 * separator whitespace the emitters write between rows.  Any other text
 * ends the cell.
 */
function groupRows(segments: NotebookSegment[]): Piece[] {
  const out: Piece[] = [];
  for (const seg of segments) {
    const prev = out.at(-1);
    if (seg.kind === "row") {
      if (prev?.kind === "cells") prev.rows.push(seg.row);
      else out.push({ kind: "cells", rows: [seg.row] });
    } else if (!(prev?.kind === "cells" && seg.text.trim() === "")) {
      out.push({ kind: "text", text: seg.text });
    }
  }
  return out;
}

/**
 * Does this text carry markup ``JMarkup`` renders rather than shows?
 *
 * ``ToolBlockView`` chooses between this component and a raw ``<pre>`` with
 * it.  It used to ask only about ``<j-``, so a notebook cell that printed
 * plain text -- ``<nb-row type="stdout" label="Out [1]:">42</nb-row>``,
 * no ``<j-*>`` block anywhere -- went to the ``<pre>`` with its tags in
 * plain sight.
 */
export function hasServerMarkup(text: string): boolean {
  return text.includes("<j-") || containsNbMarkup(text);
}

export const JMarkup = memo(function JMarkup({ text }: { text: string }) {
  const pieces = useMemo(() => groupRows(parseNbMarkup(text)), [text]);
  return (
    <div className="prose-j">
      {pieces.map((piece, i) =>
        piece.kind === "cells" ? <NotebookCells key={i} rows={piece.rows} /> : <JSegments key={i} text={piece.text} />,
      )}
    </div>
  );
});

/** Unified-diff-aware line list, used for permission prompt_lines with format_hint="diff". */
export function DiffLines({ lines }: { lines: string[] }) {
  return (
    <pre className="code-block whitespace-pre overflow-x-auto p-2">
      {lines.map((l, i) => {
        let cls = "";
        if (l.startsWith("+++") || l.startsWith("---")) cls = "diff-meta";
        else if (l.startsWith("@@")) cls = "diff-hunk";
        else if (l.startsWith("+")) cls = "diff-add";
        else if (l.startsWith("-")) cls = "diff-del";
        return <div key={i} className={cls}>{l || " "}</div>;
      })}
    </pre>
  );
}
