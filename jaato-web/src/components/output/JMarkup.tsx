/**
 * Renders a ``<j-*>`` + markdown buffer as DOM.  Pure function of its
 * ``text`` prop; memoised per block so a streaming append re-renders
 * only the block that grew.  Never uses ``innerHTML``.
 */
import { memo, useMemo } from "react";
import { parseJMarkup, type CodeLine, type Segment } from "@/protocol/jmarkup";
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
          case "pre": return <pre key={i} className="code-block surface-1 rounded-md p-2 my-1 overflow-x-auto whitespace-pre">{b.text}</pre>;
        }
      })}
    </>
  );
}

function CodeLines({ lines }: { lines: CodeLine[] }) {
  const numbered = lines.some((l) => l.n !== null);
  return (
    <pre className="code-block overflow-x-auto whitespace-pre p-2">
      {lines.map((l, i) => (
        <div key={i}>
          {numbered && <span className="ln">{l.n ?? ""}</span>}
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
    <div className="surface-1 rounded-md my-1.5 border hairline overflow-hidden">
      <div className="flex items-center justify-between px-2 py-0.5 text-[11px] text-text-muted border-b hairline">
        <span className="font-mono">{language || "text"}</span>
        <button
          type="button"
          className="hover:text-text"
          onClick={() => navigator.clipboard?.writeText(text).catch(() => undefined)}
          title="Copy code"
        >
          copy
        </button>
      </div>
      <CodeLines lines={lines} />
    </div>
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

export const JMarkup = memo(function JMarkup({ text }: { text: string }) {
  const segments = useMemo(() => parseJMarkup(text), [text]);
  return (
    <div className="prose-j">
      {segments.map((seg, i) => <SegmentView key={i} seg={seg} />)}
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
