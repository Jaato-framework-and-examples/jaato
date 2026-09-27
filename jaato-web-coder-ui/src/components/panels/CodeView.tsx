/**
 * A source file in the Files panel's content viewer (``WorkspacePanel.tsx``),
 * syntax-highlighted.  Loaded lazily with the grammars it uses
 * (``highlighter.ts``), and offered only for a path ``languageForPath``
 * names a grammar for -- anything else stays the viewer's plain text.
 *
 * **Highlighting renders React elements, never markup.**  lowlight returns
 * a syntax tree of ``span``s with ``hljs-*`` classes (``theme.css`` maps
 * them onto the ``tok-*`` palette) and text nodes; ``HastNodes`` turns that
 * tree into elements, so the file's text only ever reaches the page as
 * text.
 *
 * Highlighting is one synchronous pass over the whole file, so a file past
 * ``MAX_HIGHLIGHT_CHARS`` is shown plain rather than freezing the page;
 * the viewer says so.  Lines are numbered in a gutter, and ``wrap`` breaks
 * long lines (off by default: code is laid out in lines).
 */
import { useMemo, useState, type ReactNode } from "react";
import type { ElementContent, RootContent } from "hast";
import { lowlight } from "./highlighter";
import { languageForPath } from "@/protocol/codeLanguages";

/** Above this many characters the file is shown unhighlighted. */
export const MAX_HIGHLIGHT_CHARS = 300_000;

export interface CodeViewProps {
  text: string;
  path: string;
  /** A Tailwind max-height class for the scroll area (the viewer's expand state). */
  heightClass: string;
}

function HastNodes({ nodes }: { nodes: (RootContent | ElementContent)[] }): ReactNode {
  return nodes.map((n, i) => {
    if (n.type === "text") return n.value;
    if (n.type !== "element") return null;
    const cls = n.properties?.className;
    return (
      <span key={i} className={Array.isArray(cls) ? cls.join(" ") : typeof cls === "string" ? cls : undefined}>
        <HastNodes nodes={n.children} />
      </span>
    );
  });
}

export default function CodeView({ text, path, heightClass }: CodeViewProps) {
  const [wrap, setWrap] = useState(false);
  const lang = languageForPath(path);
  const tooBig = text.length > MAX_HIGHLIGHT_CHARS;
  const tree = useMemo(() => (lang && !tooBig && lowlight.registered(lang) ? lowlight.highlight(lang, text) : null), [lang, tooBig, text]);
  const lines = useMemo(() => {
    let n = 1;
    for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n++;
    return text.endsWith("\n") ? n - 1 : n;
  }, [text]);
  const gutter = useMemo(() => Array.from({ length: lines }, (_, i) => i + 1).join("\n"), [lines]);

  return (
    <div data-testid="code-view" data-language={tree ? lang ?? "" : ""}>
      <div className="flex items-center gap-2.5 px-2 py-1 border-b hairline text-[11px] text-text-muted">
        <span className="font-mono">{lang ?? "text"} · {lines} lines</span>
        {tooBig && <span>too large to highlight</span>}
        <button type="button" className="link text-[11px] ml-auto" onClick={() => setWrap((w) => !w)} aria-pressed={wrap}>{wrap ? "no wrap" : "wrap"}</button>
      </div>
      <div className={`${heightClass} overflow-auto flex text-[11.5px] font-mono`}>
        {!wrap && <pre aria-hidden className="m-0 py-2 pl-2 pr-2.5 text-right text-text-muted select-none border-r hairline sticky left-0 bg-surface">{gutter}</pre>}
        <pre className={`m-0 px-2 py-2 flex-1 min-w-0 ${wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre"}`}>
          <code className={tree ? `hljs language-${lang}` : undefined}>{tree ? <HastNodes nodes={tree.children} /> : text}</code>
        </pre>
      </div>
    </div>
  );
}
